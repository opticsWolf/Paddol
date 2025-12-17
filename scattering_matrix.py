# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import numpy as np
from numba import njit, prange, complex128, float64, int32

# --- Constants ---
POL_S = 0
POL_P = 1

# --- JIT Compiled Core Functions (Scalarized) ---

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

    # Linear (Default)
    if rough_type == 1: 
        factor = 1.73205080757 # sqrt(3)
        val = q * factor
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
def mod_matr_scalar(s00, s01, s10, s11):
    """
    Scalarized version of mod_matr. Converts coherent scattering matrix components 
    to intensity matrix components for incoherent light propagation.

    This function transforms the components of a 2x2 complex scattering matrix S (representing field amplitudes)
    into the components of a real-valued modified intensity matrix M that describes how intensities propagate
    in systems where wave phase information is lost, such as thick layers or with incoherent sources.

    Mathematical Definitions:

    Input:  The coherent scattering matrix relates complex fields:
        [E_out⁺]   = S * [E_in⁺]
        [E_in⁻]            [E_out⁻]

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
        s00, s01, s10, s11: Complex scalars (dtype=complex128) representing the components 
                            of the coherent scattering matrix S.

    Returns:
        m00, m01, m10, m11: Real scalars (dtype=float64) representing the components
                            of the modified intensity matrix M.

    References:
        Katsidis, C. C., & Siapkas, D. I. (2002). Applied Optics, 41(19), 3978-3987.
        Optimized Modified Matrix calculation.
    1. Removes sqrt() calls for speed (using manual norm squared).
    2. Stabilizes M[1, 1] calculation for absorbing/lossy media.
    """
    # helper: fast complex norm squared (avoids sqrt)
    mag_s00_sq = s00.real*s00.real + s00.imag*s00.imag
    mag_s01_sq = s01.real*s01.real + s01.imag*s01.imag
    mag_s10_sq = s10.real*s10.real + s10.imag*s10.imag
    
    # M00 = |1/t|^2
    m00 = mag_s00_sq
    
    # M01 = -|r_back/t|^2
    m01 = -mag_s01_sq
    
    # M10 = |r/t|^2
    m10 = mag_s10_sq
    
    # M11 calculation using determinant identity
    det_real = s00.real*s11.real - s00.imag*s11.imag - (s01.real*s10.real - s01.imag*s10.imag)
    det_imag = s00.real*s11.imag + s00.imag*s11.real - (s01.real*s10.imag + s01.imag*s10.real)
    mag_det_sq = det_real*det_real + det_imag*det_imag
    
    prod_s01s10 = s01 * s10
    mag_prod_sq = prod_s01s10.real*prod_s01s10.real + prod_s01s10.imag*prod_s01s10.imag
    
    numerator = mag_det_sq - mag_prod_sq
    
    m11 = 0.0
    if mag_s00_sq > 1e-16:
        m11 = numerator / mag_s00_sq
        
    return m00, m01, m10, m11

@njit(cache=True, inline='always')
def solve_coherent_chunk_scalar(start_idx, end_idx, n_arr, d_arr, rough_arr, 
                                rough_type_arr, lam, NSinFi, pol):
    """
    Calculates the scattering matrix components for a stack of coherent layers using Transfer Matrix Method (TMM),
    returning them as scalars to avoid array allocation overhead.

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
        s00, s01, s10, s11: Four complex scalars representing the components of the 2×2 complex scattering matrix S.
            [E⁺_out]   = S₁₁·E⁺_in + S₁₂·E⁻_out
            [E⁻_in]     S₂₁·E⁺_in + S₂₂·E⁻_out
        From this, the reflectivity R = |S₂₁/S₁₁|² and transmittivity T = 1-|r|² can be calculated.
    """
    # Initialize Identity S-Matrix scalars
    s00 = 1.0 + 0j
    s01 = 0.0 + 0j
    s10 = 0.0 + 0j
    s11 = 1.0 + 0j
    
    count = end_idx - start_idx
    two_pi_lam = (2.0 * np.pi) / lam
    
    # Loop Interfaces Backwards
    for k in range(count - 1, -1, -1):
        idx = start_idx + k
        
        # --- Interface Calculation ---
        N1 = n_arr[idx]
        N2 = n_arr[idx+1]
        sigma = rough_arr[idx+1]
        rtype = rough_type_arr[idx+1]
        
        cos1 = np.sqrt(1.0 - (NSinFi/N1)**2)
        cos2 = np.sqrt(1.0 - (NSinFi/N2)**2)
        
        # Fresnel Coefficients
        if pol == POL_P:
            num_r = N2 * cos1 - N1 * cos2
            den_r = N2 * cos1 + N1 * cos2
            num_t = 2.0 * N1 * cos1
        else: # POL_S
            num_r = N1 * cos1 - N2 * cos2
            den_r = N1 * cos1 + N2 * cos2
            num_t = 2.0 * N1 * cos1
            
        r = num_r / den_r
        t = num_t / den_r
        
        # Roughness
        if rtype == 5: # Nevot-Croce
            k0 = two_pi_lam
            kz1 = N1 * cos1
            kz2 = N2 * cos2
            nc_factor = np.exp(-2.0 * (k0 * sigma)**2 * kz1 * kz2)
            al = nc_factor
            be = nc_factor
            ga = nc_factor
        else:
            q_al = two_pi_lam * 2 * N1 * cos1 * sigma
            q_be = two_pi_lam * 2 * N2 * cos2 * sigma
            q_ga = two_pi_lam * (N1 * cos1 - N2 * cos2) * sigma
            
            al = w_function(q_al, rtype)
            be = w_function(q_be, rtype)
            ga = w_function(q_ga, rtype)
        
        # Interface Matrix Components (I)
        gat = ga * t
        inv_gat = 1.0 / gat
        
        i00 = inv_gat
        i01 = be * r * inv_gat
        i10 = al * r * inv_gat
        i11 = (ga**2 * (1.0 - r**2) + al * be * r**2) * inv_gat
        
        # Matrix Mult: S_new = I * S_old
        # Manual inline multiplication of 2x2 matrices
        _s00 = i00 * s00 + i01 * s10
        _s01 = i00 * s01 + i01 * s11
        _s10 = i10 * s00 + i11 * s10
        _s11 = i10 * s01 + i11 * s11
        
        s00, s01, s10, s11 = _s00, _s01, _s10, _s11
        
        # --- Phase Matrix Calculation ---
        if k > 0:
            d = d_arr[idx]
            beta = two_pi_lam * d * N1 * cos1
            
            if beta.imag < -50.0: beta = complex(beta.real, -50.0)
            elif beta.imag > 50.0: beta = complex(beta.real, 50.0)
            
            # P matrix is diagonal: diag(exp(-iB), exp(iB))
            p00 = np.exp(-1j * beta)
            p11 = np.exp(1j * beta)
            
            # Matrix Mult: S_new = P * S_old
            s00 = p00 * s00
            s01 = p00 * s01
            s10 = p11 * s10
            s11 = p11 * s11
            
    return s00, s01, s10, s11

@njit(parallel=True, fastmath=True, cache=True)
def core_engine_grid(wavls, sin_theta_arr, n_layers, indices_T, thicknesses, 
                     incoherent_flags, rough_types, rough_vals, 
                     calc_s, calc_p):
    """
    Computes polarized reflectance (R) and transmittance (T) spectra for a multilayer structure
    across a 2D grid of wavelengths and incident angles.

    This function handles both coherent and incoherent light propagation through layered materials
    using a hybrid matrix method that combines:
    - Transfer Matrix Method (TMM) for coherent chunks of thin layers
    - Modified intensity matrices for incoherent boundaries

    The calculation is parallelized using a flattened loop over (Angles × Wavelengths), making it 
    highly efficient for spectral-angular mapping.

    Optimizations applied:
    1. Flattened Parallel Loop: Collapses Angle x Wavelength iteration into one parallel loop.
    2. Transposed Indices: 'indices_T' is (N_wavs, N_layers), ensuring contiguous memory access 
       in the inner calculation loop.
    3. Precomputed Sine: Receives pre-calculated sin(theta) to avoid trig calls in the loop.

    Parameters:
        wavls: float array, wavelengths in [length units] to calculate spectra for.
        sin_theta_arr: float array, precomputed sine of angles of incidence.
        n_layers: int, total layers including ambient and substrate.
        indices_T: complex ndarray (n_wavs, n_layers), Transposed refractive indices. 
                   Transposed shape allows contiguous memory access in the inner loop.
        thicknesses: float array (n_layers), physical thickness of each layer [length units].
        incoherent_flags: bool array (n_layers), True for layers that break coherence.
        rough_types: int array (n_layers), roughness model type for interfaces below each layer.
        rough_vals: float array (n_layers), roughness σ values [length units] for interfaces.
        calc_s (bool): True if S-polarization should be calculated.
        calc_p (bool): True if P-polarization should be calculated.

    Returns:
        Rs_out, Rp_out: 2D float arrays [num_angles, num_wavs], reflectance for S and P polarizations.
        Ts_out, Tp_out: 2D float arrays [num_angles, num_wavs], transmittance for S and P polarizations.

    Physical Context:
    The function models real-world systems where some layers maintain coherence (thin films)
    while others break it (thick substrates). It combines coherent field calculations with
    incoherent intensity transport to accurately predict optical properties.
    """
    num_wavs = len(wavls)
    num_angles = len(sin_theta_arr)
    total_points = num_wavs * num_angles
    idx_N = n_layers - 1
    
    # Pre-allocate output arrays (Angle x Wavelength)
    # Using 'F' order (column-major) or simply matching the loop access pattern
    # Here row-major (C) is fine because we write to [a, w]
    Rs_out = np.zeros((num_angles, num_wavs), dtype=float64)
    Rp_out = np.zeros((num_angles, num_wavs), dtype=float64)
    Ts_out = np.zeros((num_angles, num_wavs), dtype=float64)
    Tp_out = np.zeros((num_angles, num_wavs), dtype=float64)
    
    # Flattened Parallel Loop
    for k in prange(total_points):
        # Decode indices
        # w varies fastest (contiguous memory access for output arrays)
        a = k // num_wavs
        w = k % num_wavs
        
        lam = wavls[w]
        
        # Optimization: Use precomputed sine
        sin_theta = sin_theta_arr[a]
        
        # Optimization: Access contiguous memory slice for this wavelength
        # indices_T is (Num_Wavs, Num_Layers)
        layer_indices = indices_T[w, :]
        
        N0 = layer_indices[0]
        NN = layer_indices[idx_N]
        NSinFi = N0 * sin_theta
        
        cos0 = np.sqrt(1.0 - (NSinFi/N0)**2)
        cosN = np.sqrt(1.0 - (NSinFi/NN)**2)
        
        # --- S POLARIZATION ---
        if calc_s:
            ms00, ms01, ms10, ms11 = 1.0, 0.0, 0.0, 1.0
            start_node = 0
            
            for i in range(n_layers):
                if incoherent_flags[i] or i == idx_N:
                    # 1. Coherent Chunk
                    # Pass the contiguous layer_indices slice
                    cs00, cs01, cs10, cs11 = solve_coherent_chunk_scalar(
                        start_node, i, layer_indices, thicknesses, 
                        rough_vals, rough_types, lam, NSinFi, POL_S)
                    
                    # 2. Mod Matrix
                    mm00, mm01, mm10, mm11 = mod_matr_scalar(cs00, cs01, cs10, cs11)
                    
                    # 3. Multiply
                    nms00 = mm00 * ms00 + mm01 * ms10
                    nms01 = mm00 * ms01 + mm01 * ms11
                    nms10 = mm10 * ms00 + mm11 * ms10
                    nms11 = mm10 * ms01 + mm11 * ms11
                    
                    ms00, ms01, ms10, ms11 = nms00, nms01, nms10, nms11
                    
                    # 4. Incoherent Prop
                    if i < idx_N:
                        N_inc = layer_indices[i]
                        d_inc = thicknesses[i]
                        cos_inc = np.sqrt(1.0 - (NSinFi/N_inc)**2)
                        
                        beta = (2.0 * np.pi * d_inc / lam) * N_inc * cos_inc
                        term1 = np.abs(np.exp(-1j * beta))**2
                        term2 = np.abs(np.exp(1j * beta))**2
                        
                        ms00 = term1 * ms00
                        ms01 = term1 * ms01
                        ms10 = term2 * ms10
                        ms11 = term2 * ms11
                    
                    start_node = i
            
            if ms00 > 1e-15:
                R_val = ms10 / ms00
                T_raw = 1.0 / ms00
                numer = np.real(NN * cosN)
                denom = np.real(N0 * cos0)
                factor = numer / denom if denom > 1e-12 else 0.0
                Rs_out[a, w] = R_val
                Ts_out[a, w] = T_raw * factor

        # --- P POLARIZATION ---
        if calc_p:
            mp00, mp01, mp10, mp11 = 1.0, 0.0, 0.0, 1.0
            start_node = 0
            
            for i in range(n_layers):
                if incoherent_flags[i] or i == idx_N:
                    cs00, cs01, cs10, cs11 = solve_coherent_chunk_scalar(
                        start_node, i, layer_indices, thicknesses, 
                        rough_vals, rough_types, lam, NSinFi, POL_P)
                    
                    mm00, mm01, mm10, mm11 = mod_matr_scalar(cs00, cs01, cs10, cs11)
                    
                    nmp00 = mm00 * mp00 + mm01 * mp10
                    nmp01 = mm00 * mp01 + mm01 * mp11
                    nmp10 = mm10 * mp00 + mm11 * mp10
                    nmp11 = mm10 * mp01 + mm11 * mp11
                    
                    mp00, mp01, mp10, mp11 = nmp00, nmp01, nmp10, nmp11
                    
                    if i < idx_N:
                        N_inc = layer_indices[i]
                        d_inc = thicknesses[i]
                        cos_inc = np.sqrt(1.0 - (NSinFi/N_inc)**2)
                        
                        beta = (2.0 * np.pi * d_inc / lam) * N_inc * cos_inc
                        term1 = np.abs(np.exp(-1j * beta))**2
                        term2 = np.abs(np.exp(1j * beta))**2
                        
                        mp00 = term1 * mp00
                        mp01 = term1 * mp01
                        mp10 = term2 * mp10
                        mp11 = term2 * mp11
                    
                    start_node = i

            if mp00 > 1e-15:
                R_val = mp10 / mp00
                T_raw = 1.0 / mp00
                numer = np.real(NN * cosN)
                denom = np.real(N0 * cos0)
                factor = numer / denom if denom > 1e-12 else 0.0
                Rp_out[a, w] = R_val
                Tp_out[a, w] = T_raw * factor
                
    return Rs_out, Rp_out, Ts_out, Tp_out

# --- Python Class Wrapper ---

class FastScatterMatrix:
    def __init__(self, layer_indices: np.ndarray, thicknesses: np.ndarray, 
                 incoherent_flags: np.ndarray, roughness_types: list[int], 
                 roughness_values: list[float], wavls: np.ndarray, theta, 
                 theta_is_radians: bool = False):
        """
        Prepares physical parameters for fast spectral-angular calculations of multilayer optical structures.

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
            roughness_types: list of int
                Integer specifying roughness model type (0-5) for each interface below a layer.
            roughness_values: list of float
                Roughness σ values [length units] for each interface below a layer.
            wavls: float ndarray
                Wavelengths at which to calculate spectra [same length units as thicknesses]
            theta: float or ndarray
                Angle(s) of incidence.
            theta_is_radians: bool
                If True, 'theta' is interpreted as radians. 
                If False (default), 'theta' is interpreted as degrees.

        Attributes:
            sin_theta_arr: Precomputed sine of angles for internal calculations
            n_layers: total number of layers including ambient and substrate
            indices_T: Transposed indices [N_wavs, N_layers] for contiguous access
            thicknesses, inc_flags: validated input arrays for core engine
            r_types, r_vals: extracted roughness parameters as contiguous arrays
        """
        self.wavls = np.ascontiguousarray(wavls, dtype=np.float64)
        
        # Handle Angle Input
        theta_arr = np.atleast_1d(theta).astype(np.float64)
        if theta_is_radians:
            theta_rads = np.ascontiguousarray(theta_arr)
        else:
            theta_rads = np.ascontiguousarray(np.radians(theta_arr))
            
        # Optimization: Precompute sine for the core engine
        self.sin_theta_arr = np.sin(theta_rads)
        self.num_angles = len(self.sin_theta_arr)
        
        self.n_layers = len(thicknesses)
        
        # Optimization: Transpose indices to (N_wavs, N_layers)
        # This makes indices[w, :] contiguous in memory, which is 
        # heavily accessed in the inner loop.
        self.indices_T = np.ascontiguousarray(layer_indices.T, dtype=np.complex128)
        
        self.thicknesses = np.ascontiguousarray(thicknesses, dtype=np.float64)
        self.inc_flags = np.ascontiguousarray(incoherent_flags, dtype=np.bool_)
        self.r_types = np.ascontiguousarray(roughness_types, dtype=np.int32)
        self.r_vals = np.ascontiguousarray(roughness_values, dtype=np.float64)
        
    def compute_RT(self, mode: str = 'u') -> dict[str, np.ndarray]:
        """
        Calculates polarized reflectance (R) and transmittance (T) spectra based 
        on requested polarization mode.
        
        This method executes the Numba-optimized `core_engine_grid` with preprocessed parameters,
        returning spectral results for S (perpendicular) or P (parallel) polarizations
        as well as unpolarized averages.
        
        Args:
            mode (str): 's', 'p', or 'u' (default).
                        'unpolarized' computes both S and P and averages them
        
        Returns:
            dict: Contains calculated spectral quantities.
                  Output arrays are 2D: [num_angles, num_wavs].
                  If only one angle was provided, the arrays are squeezed to 1D [num_wavs].
                - Rs, Rp: Reflectance for S and P polarization respectively
                - Ts, Tp: Transmittance for S and P polarization respectively
                - Ru, Tu: Unpolarized reflectance and transmittance averages
        
        Note: The computation is performed in parallel across wavelengths and angles.
        """
        calc_s = mode.lower() in ('s', 'u', 'both')
        calc_p = mode.lower() in ('p', 'u', 'both')

        Rs, Rp, Ts, Tp = core_engine_grid(
            self.wavls, self.sin_theta_arr, self.n_layers, self.indices_T, 
            self.thicknesses, self.inc_flags, self.r_types, self.r_vals,
            calc_s, calc_p
        )
        
        res = {}
        
        # Helper to squeeze if original input was scalar (legacy behavior)
        def process_output(arr):
            if self.num_angles == 1:
                return arr[0, :]
            return arr

        if calc_s:
            res['Rs'] = process_output(Rs)
            res['Ts'] = process_output(Ts)
        if calc_p:
            res['Rp'] = process_output(Rp)
            res['Tp'] = process_output(Tp)
        if calc_s and calc_p:
            res['Ru'] = process_output((Rs + Rp)/2.0)
            res['Tu'] = process_output((Ts + Tp)/2.0)
            
        return res

if __name__ == "__main__":
    import time
    
    print("--- Setting up Mock Data (Optimized Grid) ---")
    
    n_wavs = 1000
    wavls = np.linspace(300, 800, n_wavs)
    
    # Quarter-Wave Stack Design
    lambda_0 = 550.0 
    
    # Materials
    n_air = np.full(n_wavs, 1.0 + 0.0j, dtype=np.complex128)
    n_bk7 = np.full(n_wavs, 1.515 + 0.0001j, dtype=np.complex128)
    n_sio2 = np.full(n_wavs, 1.46 + 0.001j, dtype=np.complex128)
    n_ta2o5 = np.full(n_wavs, 2.10 + 0.0j, dtype=np.complex128)
    
    d_sio2 = lambda_0 / (4.0 * 1.46)
    d_ta2o5 = lambda_0 / (4.0 * 2.10)
    
    # Structure Construction
    structure = []
    structure.append([0.0, n_air, False, 0.0])
    
    num_pairs = 10
    for i in range(num_pairs):
        structure.append([d_ta2o5, n_ta2o5, False, 1])
        structure.append([d_sio2, n_sio2, False, 1])
        
    structure.append([0.0, n_bk7, False, 0.0])
    
    # Flatten Data for Class
    indices_list = []
    thick_list = []
    rough_type_list = []
    rough_value_list = []
    inc_list = []
    
    for layer in structure:
        thick_list.append(layer[0])
        indices_list.append(layer[1])
        inc_list.append(layer[2])
        r_val = layer[3]
        if r_val > 0:
            rough_type_list.append(1) 
            rough_value_list.append(r_val)
        else:
            rough_type_list.append(0) 
            rough_value_list.append(0.0)
            
    indices_arr = np.vstack(indices_list)
    thick_arr = np.array(thick_list)
    inc_arr = np.array(inc_list, dtype=np.bool_)

    # Define Angles (Scanning from 0 to 85 degrees)
    angles_deg = np.linspace(0, 46, 47) # 100 Angle steps
    
    # Instantiate
    solver = FastScatterMatrix(indices_arr, thick_arr, inc_arr, rough_type_list, 
                               rough_value_list, wavls, angles_deg, theta_is_radians=False)

    print(f"--- Computing Grid: {n_wavs} Wavelengths x {len(angles_deg)} Angles ---")
    print(f"Total Points: {n_wavs * len(angles_deg):,}")

    print("\n--- First Run (Compilation) ---")
    t_compile_start = time.time()
    _ = solver.compute_RT()
    print(f"Compilation + Exec: {time.time() - t_compile_start:.4f}s")

    print("\n--- Benchmark (100 Iterations of Full Grid) ---")
    t0 = time.time()
    for i in range(100):
        res = solver.compute_RT()
    
    dt = time.time() - t0
    total_ops = (n_wavs * len(angles_deg)) * 100
    print(f"Total time: {dt:.6f}s")
    print(f"Speed: {total_ops / dt:.2f} points/sec")

    # --- 5. Display Sample Results ---
    # We select a specific angle (e.g., 45 degrees) to display the spectral data
    display_angle = 45.0
    idx_ang = np.abs(angles_deg - display_angle).argmin()
    actual_ang = angles_deg[idx_ang]

    # Extract 1D slices for the print loop
    # res['Rs'] is (Angles, Wavelengths), so we slice [idx_ang, :]
    Rs = res['Rs'][idx_ang, :]
    Rp = res['Rp'][idx_ang, :]
    
    # Get Average Transmission (Tu is already calculated by default mode='u')
    if 'Tu' in res:
        T_avg = res['Tu'][idx_ang, :]
    else:
        # Fallback if mode was changed
        T_avg = (res['Ts'][idx_ang, :] + res['Tp'][idx_ang, :]) / 2.0

    print(f"\n--- Sample Results (Angle={actual_ang:.1f}deg, Every 100th point) ---")
    print(f"{'Wavel (nm)':<12} | {'Rs':<10} | {'Rp':<10} | {'Trans (Avg)':<10}")
    print("-" * 50)
    for i in range(0, n_wavs, 100):
        print(f"{wavls[i]:<12.1f} | {Rs[i]:<10.5f} | {Rp[i]:<10.5f} | {T_avg[i]:<10.5f}")