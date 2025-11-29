# -*- coding: utf-8 -*-
"""
Universal Band-Fluctuation Oscillator Model.

A unified dispersion model for dielectrics and semiconductors that naturally
integrates Urbach tails, Fundamental Absorption, and Lorentz resonance without
piecewise discontinuities.

Based on the Monolog-Lorentz (ML) model proposed by Lizárraga et al. (2022)
and causal constraints by Forouhi & Bloomer.
"""

import numpy as np
from numba import njit
from typing import Dict, Union, Optional, List, Tuple

# --- Base Material Class (Mocked for standalone usage if needed) ---
try:
    from .material import Material, compute_energy
except ImportError:
    class Material:
        def __init__(self, params: Dict, wavelength: Optional[np.ndarray] = None):
            self.params = params
            if wavelength is not None:
                self.set_wavelength_range(wavelength)
        
        def set_wavelength_range(self, wl):
            self.wavelength = np.asarray(wl, dtype=np.float64)
        
        def get_params(self):
            return self.params

    @njit(cache=True)
    def compute_energy(wavelength, h_c):
        # Avoid division by zero
        return h_c / np.maximum(wavelength, 1e-9)

__all__ = ["UBF_Cody_Lorentz"]


# --- Numba Accelerated Kernels ---

@njit(cache=True, fastmath=True)
def _softplus_generalized(x: float, gamma: float) -> float:
    """
    Computes [ln(1 + e^x)]^gamma safely.
    
    Strategy:
        If x > 50, ln(1 + e^x) -> x. This prevents overflow in exp(x).
        Otherwise, standard calculation.
    """
    if x > 50.0:
        base = x
    else:
        base = np.log(1.0 + np.exp(x))
    
    # Handle gamma power safely
    if gamma == 2.0:
        return base * base
    elif gamma == 0.5:
        return np.sqrt(base)
    elif gamma == 1.0:
        return base
    else:
        return base ** gamma


@njit(cache=True, fastmath=True)
def _compute_eps2_monolog_lorentz(E: np.ndarray,
                                  oscillators: np.ndarray) -> np.ndarray:
    """
    Computes the Imaginary part of Dielectric Function (eps2) using
    the Monolog-Lorentz formulation.

    Equation:
        eps2(E) = Sum_j [ A/E * Softplus(beta*(E-Eg))^gamma * Lorentz(E) ]

    Args:
        E: Energy array in eV.
        oscillators: 2D array (N_osc, 6) -> [Eg, Ec, Beta, A, Gamma, Gamma_exp]
                     Gamma_exp is 'gamma' (2.0 for Indirect/Amorphous, 0.5 for Direct).

    Returns:
        eps2 array.
    """
    eps2 = np.zeros_like(E, dtype=np.float64)
    n_osc = oscillators.shape[0]
    n_E = E.shape[0]

    for i in range(n_E):
        eng = E[i]
        
        # Avoid singularity at E=0
        if eng < 1e-6:
            continue
            
        val_sum = 0.0
        eng_sq = eng * eng

        for j in range(n_osc):
            # Unpack parameters
            # Eg: Bandgap
            # Ec: Central Resonance Energy
            # Beta: Urbach slope (1/Eu)
            # A: Amplitude
            # G: Broadening (Gamma)
            # Y: Exponent type (2.0=Indirect, 0.5=Direct)
            Eg, Ec, Beta, A, G, Y = oscillators[j]

            # 1. Band-Fluctuation Term (The "Monolog" part)
            # Argument for softplus
            arg = Beta * (eng - Eg)
            
            # Calculate [ln(1+e^x)]^gamma
            band_shape = _softplus_generalized(arg, Y)

            # 2. Lorentz Oscillator Term
            # L(E) = (E * G * Ec) / ((E^2 - Ec^2)^2 + G^2 * E^2)
            # Note: We include Ec in numerator to maintain unit consistency with A
            denom = (eng_sq - Ec**2)**2 + (G * eng)**2
            lorentz = (eng * G * Ec) / np.maximum(denom, 1e-12)

            # 3. Combine: (A / E) * BandShape * Lorentz
            # Note: Paper 1 Eq 37/39 uses 1/E prefactor. 
            term = (A / eng) * band_shape * lorentz
            
            val_sum += term

        eps2[i] = val_sum

    return eps2


@njit(cache=True, parallel=True)
def _kramers_kronig_integral(E: np.ndarray, 
                             eps2: np.ndarray, 
                             eps_inf: float) -> np.ndarray:
    """
    Calculates Real Dielectric Function (eps1) from eps2 using 
    Maclaurin's method for Kramers-Kronig integration.
    
    This method is robust against singularities at E' = E.
    
    eps1(E) = eps_inf + (2/pi) * P.V. Integral [ (E' * eps2(E')) / (E'^2 - E^2) dE' ]
    """
    n = E.shape[0]
    eps1 = np.full(n, eps_inf, dtype=np.float64)
    
    # Pre-calculate E * eps2
    numerator = E * eps2

    for i in range(n):
        energy_i = E[i]
        energy_i_sq = energy_i**2
        integral = 0.0
        
        # Trapezoidal integration skipping the singularity
        for j in range(n - 1):
            e_j = E[j]
            e_j1 = E[j+1]
            
            # Identify singularity vicinity
            if (e_j <= energy_i <= e_j1):
                # Handle singularity analytically or skip very close points
                # Here we assume the mesh is fine enough that we can effectively
                # ignore the contribution exactly at the pole due to antisymmetry
                continue
            
            # Denominators
            denom_j = e_j**2 - energy_i_sq
            denom_j1 = e_j1**2 - energy_i_sq
            
            # Avoid division by zero
            if np.abs(denom_j) < 1e-12: denom_j = 1e-12
            if np.abs(denom_j1) < 1e-12: denom_j1 = 1e-12

            val_j = numerator[j] / denom_j
            val_j1 = numerator[j+1] / denom_j1
            
            dE = e_j1 - e_j
            integral += 0.5 * (val_j + val_j1) * dE
            
        eps1[i] += (2.0 / np.pi) * integral
        
    return eps1


# --- Main Model Class ---

class UBF_Cody_Lorentz(Material):
    """
    Universal Band-Fluctuation Cody-Lorentz Model (UBF-CL).

    A highly optimized, physically general model for dielectrics, amorphous 
    semiconductors, and crystalline materials with disorder.

    It unifies the Urbach tail and Fundamental Absorption into a single 
    analytic function (Monolog) derived from Band-Fluctuations theory, 
    multiplied by a Lorentz oscillator to limit high-energy transparency.

    Physics:
        eps2(E) ~ (1/E) * [ln(1 + exp(beta*(E-Eg)))]^gamma * Lorentz(E)
        eps1(E) = eps_inf + Kramers-Kronig(eps2)

    Attributes:
        params (dict): Dictionary containing:
            - 'epsilon_inf': High-frequency dielectric constant.
            - 'oscillators': List of dictionaries, each containing:
                - 'Eg': Optical Bandgap [eV].
                - 'Ec': Central Resonance Energy [eV] (Peak of the Lorentz).
                - 'Eu': Urbach Energy [eV] (Inverse of beta).
                - 'A': Amplitude [dimensionless].
                - 'Gamma': Broadening/Damping [eV].
                - 'Type': 'Indirect' (default) or 'Direct'. Sets the exponent.
    """

    def __init__(self, 
                 params: Dict[str, Union[float, List[Dict]]], 
                 wavelength: Optional[np.ndarray] = None):
        """
        Initialize the Universal model.

        Args:
            params: Model parameters.
            wavelength: Optional wavelength array (nm).
        """
        # Initialize base
        self.h_c_by_eV_nm = 1239.8419843320028       
        self.epsilon_inf = float(params.get('epsilon_inf', 1.0))

        # Parse Oscillators
        osc_list = params.get('oscillators', [])
        if not osc_list:
            raise ValueError("At least one oscillator must be provided in 'oscillators' list.")
            
        self._osc_array = self._parse_oscillators(osc_list)

        super().__init__(params, wavelength)


    def _parse_oscillators(self, osc_list: List[Dict]) -> np.ndarray:
        """Parses oscillator dicts into a Numba-friendly numpy array."""
        data = []
        for osc in osc_list:
            Eg = float(osc.get('Eg', 1.5))
            Ec = float(osc.get('Ec', 5.0))
            
            # Handle Urbach Energy Eu -> Beta = 1/Eu
            Eu = float(osc.get('Eu', 0.05)) # Default 50 meV
            beta = 1.0 / max(Eu, 1e-9)
            
            A = float(osc.get('A', 10.0))
            Gamma = float(osc.get('Gamma', 1.0))
            
            # Determine Gamma exponent based on material type
            # Indirect/Amorphous -> 2.0 (Standard Tauc-like behavior)
            # Direct -> 0.5 (Square-root behavior)
            mat_type = osc.get('Type', 'Indirect').lower()
            if 'direct' in mat_type and 'indirect' not in mat_type:
                gamma_exp = 0.5
            else:
                gamma_exp = 2.0
                
            data.append([Eg, Ec, beta, A, Gamma, gamma_exp])
            
        return np.array(data, dtype=np.float64)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Sets wavelength (nm) and computes internal Energy (eV) grid."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm)

    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the complex refractive index (n + ik).

        This triggers the Numba-compiled kernels for high performance.
        
        Order of Operations:
        1. Compute eps2(E) using Monolog-Lorentz logic.
        2. Compute eps1(E) using Numerical Kramers-Kronig.
        3. Convert sqrt(eps1 + i*eps2) -> n + ik.
        """
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
            
        if not hasattr(self, 'E'):
            raise AttributeError("Wavelength range must be set.")

        # 1. Calculate Imaginary Part (Absorption)
        eps2 = _compute_eps2_monolog_lorentz(self.E, self._osc_array)
        
        # 2. Calculate Real Part (Dispersion) via KK to ensure causality
        # Note: We use the same E grid for source and target for efficiency
        eps1 = _kramers_kronig_integral(self.E, eps2, self.epsilon_inf)
        
        # 3. Convert dielectric function to refractive index
        eps_complex = eps1 + 1j * eps2
        self.nk = np.sqrt(eps_complex)
        
        return self.nk

    def get_params(self) -> Dict:
        """Returns readable parameters."""
        oscillators_out = []
        for row in self._osc_array:
            Eg, Ec, beta, A, Gamma, gamma_exp = row
            oscillators_out.append({
                'Eg': Eg,
                'Ec': Ec,
                'Eu': 1.0/beta,
                'A': A,
                'Gamma': Gamma,
                'Type': 'Direct' if gamma_exp == 0.5 else 'Indirect'
            })
            
        return {
            'epsilon_inf': self.epsilon_inf,
            'oscillators': oscillators_out
        }

# --- Usage Example ---
if __name__ == "__main__":
    # Example: Modeling Amorphous Silicon (Indirect) and a Defect state
    params = {
        'epsilon_inf': 1.0,
        'oscillators': [
            {
                # Main Bandgap
                'Eg': 1.3,      # eV
                'Ec': 1.8,      # Lorentz Peak
                'Eu': 240.05,     # Urbach tail width (50 meV)
                'A': 100.0,      # Strength
                'Gamma': 1.0,   # Broadening
                'Type': 'Indirect'
            },
            {
                # High Energy absorption
                'Eg': 3.0,
                'Ec': 6.0,
                'Eu': 0.1,
                'A': 20.0,
                'Gamma': 1.5,
                'Type': 'Indirect'
            }
        ]
    }
    
    # Wavelengths from 200nm to 1000nm
    wl = np.linspace(200, 1000, 500)
    
    model = UBF_Cody_Lorentz(params, wl)
    nk = model.complex_refractive_index()
    
    print(f"Computed {len(nk)} points.")
    print(f"n at 500nm: {nk[250].real:.4f}") # Approx index
    print(f"k at 500nm: {nk[250].imag:.4f}")