# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 00:09:09 2025

@author: Frank
Updated for Performance, Strict Typing, and Documentation
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from ema_models_v1 import looyenga_eps

# Define standardized types for Numba compatibility
FLOAT_TYPE = np.float64
COMPLEX_TYPE = np.complex128
INT_TYPE = np.int32

class Layer:
    """
    Represents a layer in a thin film structure with optimized memory usage.
    
    A layer is characterized by its material properties, thickness, and potential imperfections like
    roughness. It can model both homogeneous and inhomogeneous (graded) layers. The class is designed for
    memory efficiency using __slots__ when used in large structures or optimization routines.
    
    Uses __slots__ for faster attribute access and reduced memory footprint.

    Attributes:
        material: Name of the material defining optical constants.
        coherent: Whether light propagation should be treated as coherent through this layer.
        roughness: Surface roughness (RMS) in Angstroms that causes scattering.
        rough_type: Integer code specifying which roughness model to use (e.g., Neede, Gaussian).
        _inhomogen: Whether the refractive index varies with depth (graded layer).
        inh_delta: Relative deviation range of refractive index in graded layers.
        interface: Whether this layer has an abrupt interface with the next layer.
        interface_thickness: Thickness of mixed-material transition zone at boundaries.
        thickness: Physical thickness of the layer in nanometers.
    """
    __slots__ = (
        'material', 'coherent', '_inhomogen', 'rough_type', '_inh_delta',
        'roughness', 'interface', 'interface_thickness', '_thickness',
        'optimize', 'needle', 'layer_typ', 'mask', 'sub_layer_count'
    )

    def __init__(
        self,
        thickness: float = 1.0,
        material_name: str = '',
        coherent: bool = True,
        roughness: float = 0.0,
        rough_type: int = 0,
        inhomogen: bool = False,
        inh_delta: float = 0.1,
        interface: bool = False,
        interface_thickness: float = 0.0,
        optimize: bool = True,
        needle: bool = True,
        layer_typ: int = 1
    ) -> None:
        """Initializes a thin film layer with specified physical and optical properties.

        Args:
            thickness: Physical thickness in nanometers (default: 1.0).
            material_name: String identifier for the material's optical constants.
            coherent: If True, light is treated as coherent through this layer; otherwise incoherent (default: True).
            roughness: Surface roughness in Angstroms that causes light scattering (default: 0.0).
            rough_type: Integer code specifying the roughness model to use (default: 0 for none).
            inhomogen: If True, models a graded layer where refractive index varies with depth.
            inh_delta: Relative deviation range of refractive index for graded layers (default: 0.1).
            interface: If True, creates an abrupt transition zone between this and next layer.
            interface_thickness: Thickness in nanometers of the mixed-material transition zone (default: 0.0).
            optimize: Whether this layer's parameters should be included in optimization routines (default: True).
            needle: Whether to use Needle roughness model if applicable (default: True for rough_type > 0).
            layer_typ: Integer code specifying special handling cases like substrates.
        """
        self.material: str = material_name
        self.coherent: bool = coherent
        self._inhomogen: bool = inhomogen
        self.rough_type: int = rough_type
        self._inh_delta: float = inh_delta
        self.roughness: float = roughness
        self.interface: bool = interface
        self.interface_thickness: float = interface_thickness
        self._thickness: float = float(thickness)
        self.optimize: bool = optimize
        self.needle: bool = needle
        self.layer_typ: int = layer_typ

        self._initialize_mask()
        self._refine_layer_count()

    def __call__(self) -> Tuple[str, float]:
        """Returns a tuple containing the material name and thickness of this layer.

        Returns:
            Tuple[str, float]: (material_name, thickness_in_nm)
        """
        return (self.material, self._thickness)

    @property
    def thickness(self) -> float:
        """Gets the physical thickness of the layer in nanometers.

        When inhomogeneous layers are present, this is the total thickness before subdivision.

        Returns:
            float: Layer thickness in nanometers.
        """
        return self._thickness
    
    @thickness.setter
    def thickness(self, value: float) -> None:
        """Sets the physical thickness of the layer in nanometers.

        Args:
            value: Desired thickness in nanometers (must be non-negative).
        
        Side Effects:
            If the layer is inhomogeneous, this triggers automatic recalculation of sub-layer count.
        """
        self._thickness = float(value)
        if self._inhomogen:
            self._refine_layer_count()
    
    @property
    def inhomogen(self) -> bool:
        """Gets whether this is an inhomogeneous (graded index) layer.

        Returns:
            bool: True if refractive index varies with depth.
        """
        return self._inhomogen
    
    @inhomogen.setter
    def inhomogen(self, value: bool) -> None:
        """Sets the inhomogeneity flag for this layer.

        Args:
            value: If True, models a graded refractive index profile.
        
        Side Effects:
            Triggers recalculation of sub-layer count if enabled.
        """
        self._inhomogen = bool(value)
        if self._inhomogen:
            self._refine_layer_count()
    
    @property
    def inh_delta(self) -> float:
        """Gets the relative deviation range for graded refractive index profiles.

        Returns:
            float: Fractional deviation from mean refractive index.
        """
        return self._inh_delta
    
    @inh_delta.setter
    def inh_delta(self, value: float) -> None:
        """Sets the inhomogeneity delta parameter.

        Args:
            value: Relative deviation range (typically 0-1).
        
        Side Effects:
            Triggers recalculation of sub-layer count if inhomogeneous.
        """
        self._inh_delta = float(value)
        if self._inhomogen:
            self._refine_layer_count()

    def _initialize_mask(self) -> None:
        """Initializes the optimization mask array based on current layer flags.

        The mask is a bitvector indicating which parameters should be optimized, with positions
        corresponding to: [thickness, coherent, inhomogeneous, roughness_model].
        """
        # [thickness_coherent, inhomogen, roughness]
        self.mask = np.array([
            1, 
            int(self.coherent), 
            int(self._inhomogen), 
            1 if self.rough_type > 0 else 0
        ], dtype=INT_TYPE)

    def _refine_layer_count(self) -> None:
        """
        Calculates the required number of sub-layers for inhomogeneous gradients.
        
        Uses a power-law heuristic: count ~ ceil(d^0.4 * (1 + 5*delta)) where d is thickness in nm,
        and delta is the inhomogeneity factor. This balances numerical accuracy with computational cost.

        The formula ensures thicker layers have more subdivisions, while larger inhomogeneities also
        require finer discretization.
        """
        if self._inhomogen and self._thickness > 0:
            factor = 1.0 + (self._inh_delta / 0.1) * 0.5
            self.sub_layer_count = int(np.ceil(self._thickness ** 0.4) * factor) + 1
        else:
            self.sub_layer_count = 1

    def get_properties(self) -> Dict[str, Any]:
        """Returns a dictionary of all layer properties for serialization or UI display.

        Returns:
            Dict[str, Any]: Key-value pairs containing all current state.
        """
        return {
            'thickness': self._thickness,
            'material': self.material,
            'coherent': self.coherent,
            'inhomogen': self._inhomogen,
            'inh_delta': self._inh_delta,
            'rough_type': self.rough_type,
            'roughness': self.roughness,
            'interface': self.interface,
            'interface_thickness': self.interface_thickness,
            'optimize': self.optimize,
            'needle': self.needle,
            'mask': self.mask
        }

    def set_properties(self, properties: Dict[str, Any]) -> None:
        """Updates layer properties from a dictionary of key-value pairs.

        Args:
            properties: Dictionary containing parameters to update.
                      Only keys matching existing attributes are updated.

        Side Effects:
            If interface or inhomogeneity is enabled, triggers recalculation of sub-layer count.
        """
        for key, value in properties.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        if self.interface or self._inhomogen:
            self._refine_layer_count()

    def clone(self) -> 'Layer':
        """
        Creates a high-performance deep copy of the Layer.
        
        Bypasses __init__ for speed and manually copies all slots.
        Essential for optimization algorithms (Genetic/Evolutionary) that
        need to spawn populations of layers rapidly.
        """
        obj = Layer.__new__(Layer)
        obj.material = self.material
        obj.coherent = self.coherent
        obj._inhomogen = self._inhomogen
        obj.rough_type = self.rough_type
        obj._inh_delta = self._inh_delta
        obj.roughness = self.roughness
        obj.interface = self.interface
        obj.interface_thickness = self.interface_thickness
        obj._thickness = self._thickness
        obj.optimize = self.optimize
        obj.needle = self.needle
        obj.layer_typ = self.layer_typ
        obj.sub_layer_count = self.sub_layer_count
        
        # NumPy arrays must be explicitly copied
        obj.mask = self.mask.copy()
        
        return obj

    def __repr__(self) -> str:
        """String representation for easier debugging."""
        return (f"Layer(mat='{self.material}', d={self._thickness:.2f}nm, "
                f"rough={self.roughness:.2f}A, opt={self.optimize})")


class Group:
    """Represents a group of materials sharing common optical and manufacturing properties.

    This allows applying consistent error models and scaling factors to multiple layers. For example,
    a "SiO2" group might define how thickness varies across wafers due to deposition processes.

    Attributes:
        group_name: String identifier for the material family.
        thick_factor: Multiplicative factor applied to layer thicknesses (default: 1.0).
        n_factor/k_factor: Factors scaling real/imaginary parts of refractive indices.
        error_mask: Bitmask indicating which parameters are subject to manufacturing errors.
    """
    __slots__ = (
        'group_name', 'thick_factor', 'thick_summand', 'n_factor', 'k_factor',
        'inh_delta_summand', 'roughness_summand', 'interface_summand',
        'error_mask', 'optimization_mask',
        'thickness_error_type', 'n_error_type', 'k_error_type',
        'inh_delta_error_type', 'roughness_error_type', 'interface_error_type',
        'thickness_error_params', 'inh_delta_error_params', 'roughness_error_params',
        'interface_error_params', 'n_error_params', 'k_error_params'
    )

    def __init__(
        self,
        group_name: str,
        thick_factor: float = 1.0,
        thick_summand: float = 0.0,
        n_factor: float = 1.0,
        k_factor: float = 0.0,
        inh_delta_summand: float = 0.0,
        roughness_summand: float = 0.0,
        interface_summand: float = 0.0
    ) -> None:
        """Initializes a material group with shared optical and error parameters.

        Args:
            group_name: String identifier for the material group.
            thick_factor: Multiplicative factor applied to all thicknesses (default: 1.0).
            thick_summand: Additive constant in thickness (nm, default: 0.0).
            n_factor: Scaling factor for real part of refractive index (n, default: 1.0).
            k_factor: Scaling factor for imaginary part of refractive index (k, default: 0.0).
            inh_delta_summand: Constant added to inhomogeneity deltas (fractional, default: 0.0).
            roughness_summand: Additive constant to surface roughness (Angstroms, default: 0.0).
            interface_summand: Added to interface thickness (nm, default: 0.0).
        """
        self.group_name = group_name
        self.thick_factor = thick_factor
        self.thick_summand = thick_summand
        self.n_factor = n_factor
        self.k_factor = k_factor
        self.inh_delta_summand = inh_delta_summand
        self.roughness_summand = roughness_summand
        self.interface_summand = interface_summand
        
        self.error_mask = [0] * 6 
        self.optimization_mask = [0] * 7  

        self.thickness_error_type = 0 
        self.n_error_type = 0
        self.k_error_type = 0
        self.inh_delta_error_type = 0
        self.roughness_error_type = 0
        self.interface_error_type = 0
        
        default_params = {
            'abs_mean_delta_g': 0.0, 'abs_std_dev': 0.01,
            'rel_mean_delta_g': 0.0, 'rel_std_dev': 1.0,         
            'abs_mean_delta_h': 0.0, 'abs_variance': 0.01,
            'rel_mean_delta_h': 0.0, 'rel_variance': 1.0
        }

        self.thickness_error_params = default_params.copy()
        self.inh_delta_error_params = default_params.copy()
        self.roughness_error_params = default_params.copy()
        self.interface_error_params = default_params.copy()
        self.n_error_params = default_params.copy()
        self.k_error_params = default_params.copy()

    @property
    def nk_factor(self) -> complex:
        """Returns the combined refractive index scaling factor as a complex number.

        Returns:
            complex: (n_factor + i*k_factor), representing scaling for real/imaginary parts.
        """
        return complex(self.n_factor, self.k_factor)

    def _apply_error(self, value: float, error_type: int, error_params: dict) -> float:
        """
        Applies a stochastic manufacturing error to a value based on distribution settings.
        Supports three distribution types for simulating manufacturing variability.
        
        Args:
            value (float): The base value to perturb.
            error_type (int): 
                0 = Gaussian (Normal) distribution.
                1 = Uniform distribution.
                2 = Combined (Gaussian + Uniform).
            error_params (dict): Dictionary containing statistical parameters:
                - *_mean_delta_g, *_std_dev: Gaussian parameters (abs/rel).
                - *_variance: Uniform bounds (abs/rel).

        Returns:
            float: The perturbed value.
        """
        if error_type == 0: # Gaussian
            abs_err = np.random.normal(error_params['abs_mean_delta_g'], error_params['abs_std_dev'])
            rel_err = np.random.normal(error_params['rel_mean_delta_g'], error_params['rel_std_dev']) * value
            return value + abs_err + rel_err
        
        if error_type == 1: # Uniform
            abs_err = np.random.uniform(-error_params['abs_variance'], error_params['abs_variance'])
            rel_err = np.random.uniform(-error_params['rel_variance'], error_params['rel_variance']) * value
            return value + abs_err + rel_err

        if error_type == 2: # Combined
            g_abs = np.random.normal(error_params['abs_mean_delta_g'], error_params['abs_std_dev'])
            g_rel = np.random.normal(error_params['rel_mean_delta_g'], error_params['rel_std_dev']) * value
            u_abs = np.random.uniform(-error_params['abs_variance'], error_params['abs_variance'])
            u_rel = np.random.uniform(-error_params['rel_variance'], error_params['rel_variance']) * value
            return value + g_abs + g_rel + u_abs + u_rel
            
        return value

    def thickness_error(self, value: float) -> float:
        """Apply thickness error; result clamped to >= 0.
        
        Args:
            value: Nominal thickness in nanometers.
        
        Returns:
            float: Thickness perturbed by errors, clamped to >= 0 nm.
        """
        return max(0.0, self._apply_error(value, self.thickness_error_type, self.thickness_error_params))

    def inh_delta_error(self, value: float) -> float:
        """Apply inhomogeneity delta error.
        
        Args:
            value: Nominal inhomogeneity delta (fractional).

        Returns:
            float: Perturbed inhomogeneity delta.
        """
        
        return self._apply_error(value, self.inh_delta_error_type, self.inh_delta_error_params)

    def sr_roughness_error(self, value: float, thickness: float) -> float:
        """Apply roughness error; result clamped to >= 0.
        Args:
            value: Nominal surface roughness in Angstroms.
            thickness: Layer thickness (used for relative scaling).

        Returns:
            float: Roughness perturbed by errors, clamped to >= 0 A.
        """
        return max(0.0, self._apply_error(value, self.roughness_error_type, self.roughness_error_params))

    def interface_error(self, value: float, thickness: float) -> float:
        """Apply interface thickness error; result clamped to >= 0.
        
        Args:
            value: Nominal interface thickness in nanometers.
            thickness: Total layer thickness (used for relative scaling).

        Returns:
            float: Interface thickness perturbed by errors, clamped to >= 0 and <= total thickness.
        """
        return max(0.0, self._apply_error(value, self.interface_error_type, self.interface_error_params))

    def nk_error(self, nk_value: complex) -> complex:
        """Apply errors independently to real (n) and imaginary (k) parts.
        
        Args:
        nk_value: Complex number where .real is refractive index (n), .imag is extinction coefficient (k).

        Returns:
            complex: Perturbed complex refractive index with non-negative real part.
            (Negative n values are clamped to 0 which may cause numerical instability warnings)
        """
        n_val = self._apply_error(nk_value.real, self.n_error_type, self.n_error_params)
        k_val = self._apply_error(nk_value.imag, self.k_error_type, self.k_error_params)
        return complex(max(0.0, n_val), k_val)

    def get_properties(self) -> Dict[str, Any]:
        """Returns a comprehensive dictionary containing all properties of the material group.
    
        This method is designed for serialization and user interface display. The returned dictionary
        includes all scalar parameters, array-based masks, and nested error parameter dictionaries that
        define this group's behavior in manufacturing simulations.
    
        Returns:
            Dict[str, Any]: A dictionary with keys corresponding to all instance attributes.
                Contains the following top-level keys:
                    - Scalar factors: 'group_name', 'thick_factor', 'n_factor', etc.
                    - Bitmask arrays: 'error_mask' and 'optimization_mask'
                    - Error model dictionaries: Each contains statistical parameters for a specific parameter type
                      (e.g., thickness, refractive index components)
        """
        return {
            'group_name': self.group_name,
            'thick_factor': self.thick_factor,
            'thick_summand': self.thick_summand,
            'n_factor': self.n_factor,
            'k_factor': self.k_factor,
            'inh_delta_summand': self.inh_delta_summand,
            'roughness_summand': self.roughness_summand,
            'interface_summand': self.interface_summand,
            'error_mask': self.error_mask,
            'optimization_mask': self.optimization_mask,
            'thickness_error_type': self.thickness_error_type,
            'n_error_type': self.n_error_type,
            'k_error_type': self.k_error_type,
            'inh_delta_error_type': self.inh_delta_error_type,
            'roughness_error_type': self.roughness_error_type,
            'interface_error_type': self.interface_error_type,
            'thickness_error_params': self.thickness_error_params,
            'inh_delta_error_params': self.inh_delta_error_params,
            'roughness_error_params': self.roughness_error_params,
            'interface_error_params': self.interface_error_params,
            'n_error_params': self.n_error_params,
            'k_error_params': self.k_error_params,
        }

    def set_properties(self, properties: Dict[str, Any]) -> None:
        """Bulk updates multiple group properties from a dictionary of key-value pairs.

        This method provides a bulk update interface that safely modifies only the attributes that exist
        in this class instance. It is primarily used for configuration loading and UI updates where
        individual property setters might be impractical.
    
        Args:
            properties: Dictionary containing parameter names (keys) and their desired values.
                        Only existing attributes are updated; others are ignored silently.
    
        Note:
            This method does not validate parameter ranges or relationships. Callers should ensure
            that the provided dictionary contains physically meaningful values for the application context.
    
        Example:
            group.set_properties({
                'thick_factor': 1.05,
                'n_factor': 0.98,
                'k_factor': 0.02
            })
        """
        for key, value in properties.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def clone(self) -> 'Group':
        """Creates a high-performance deep copy of the Group instance.
        
        Returns:
            Group: A new instance with identical properties to self.
        """
        obj = Group.__new__(Group)
        obj.group_name = self.group_name
        
        # Copy scalar factors
        obj.thick_factor = self.thick_factor
        obj.thick_summand = self.thick_summand
        obj.n_factor = self.n_factor
        obj.k_factor = self.k_factor
        obj.inh_delta_summand = self.inh_delta_summand
        obj.roughness_summand = self.roughness_summand
        obj.interface_summand = self.interface_summand
        
        # Copy Lists/Dicts (Mutable types)
        obj.error_mask = self.error_mask[:]
        obj.optimization_mask = self.optimization_mask[:]
        
        obj.thickness_error_type = self.thickness_error_type
        obj.n_error_type = self.n_error_type
        obj.k_error_type = self.k_error_type
        obj.inh_delta_error_type = self.inh_delta_error_type
        obj.roughness_error_type = self.roughness_error_type
        obj.interface_error_type = self.interface_error_type
        
        obj.thickness_error_params = self.thickness_error_params.copy()
        obj.inh_delta_error_params = self.inh_delta_error_params.copy()
        obj.roughness_error_params = self.roughness_error_params.copy()
        obj.interface_error_params = self.interface_error_params.copy()
        obj.n_error_params = self.n_error_params.copy()
        obj.k_error_params = self.k_error_params.copy()
        
        return obj

    def __repr__(self) -> str:
        """String representation for easier debugging.

        Returns:
            str: A concise string showing key properties.
        """
        return f"Group(name='{self.group_name}', thick_factor={self.thick_factor:.3f})"


class TF_Structure:
    """
    Manages the translation of high-level layer definitions into numerical arrays
    compatible with the FastScatterMatrix solver.
    
    This class handles the complex process of converting logical layers (which may include interfaces,
    graded refractive indices, and roughness models) into the flat arrays required by the FastScatterMatrix
    solver. Key responsibilities include:

    1. Expanding inhomogeneous layers into multiple sub-layers with appropriate index gradients.
    2. Calculating effective refractive indices for interface zones using Looyenga mixing model.
    3. Generating columnar arrays in Structure-of-Arrays format required by the solver.

    Attributes:
        layer_list: List of high-level Layer objects defining the structure.
        group_dict: Optional mapping from material names to Group error models.
        active_material_dict: Dictionary providing optical constants for materials.
    """

    def __init__(self, layer_list: List[Layer], group_dict: Optional[Dict[str, Group]] = None, 
                 active_material_dict: Optional[Dict[str, Any]] = None):
        """Initializes the thin film structure model with layer and material definitions.

        Args:
            layer_list: Ordered list of Layer objects defining the stack.
            group_dict: Optional dict mapping material names to Group error models (default: {}).
            active_material_dict: Dict providing optical constants for each material by name.
                                  Must contain all materials referenced in layers.
        """
        self.layer_list = layer_list
        self.group_dict = group_dict or {}
        self.active_material_dict = active_material_dict or {}
        # We hold the "simple" representation for reference if needed
        self.simple_layer_list: List[List[Any]] = [] 

    def generate_simple_layer_list(self) -> List[List[Any]]:
        """Generates a legacy-style representation as a list of lists (deprecated but preserved).

        This is primarily maintained for backward compatibility with older code that expects
        the format: [[thickness, nk_value, coherent_flag, roughness, rough_type], ...]

        Returns:
            List[List[Any]]: The simple layer representation. Also stored in self.simple_layer_list.
        """
        self.get_solver_inputs() 
        return self.simple_layer_list

    def validate(self) -> List[str]:
        """Validates physical constraints on all layers in the structure.

        Checks include:
        - Non-negative thicknesses and roughness values
        - Interface thickness <= layer thickness
        - Existence of materials in the provided dictionary

        Returns:
            List[str]: Error messages for any violations. Empty list means valid.
        """
        errors = []
        if not self.layer_list:
            errors.append("Structure contains no layers.")
            return errors
            
        for i, layer in enumerate(self.layer_list):
            # 1. Physical Constraints
            if layer.thickness < 0:
                errors.append(f"Layer {i} ({layer.material}): Negative thickness {layer.thickness} nm.")
            if layer.roughness < 0:
                errors.append(f"Layer {i} ({layer.material}): Negative roughness {layer.roughness} A.")
                
            # 2. Interface Constraints
            if layer.interface and layer.interface_thickness >= layer.thickness:
                errors.append(f"Layer {i} ({layer.material}): Interface thickness ({layer.interface_thickness}) "
                              f"cannot exceed total layer thickness ({layer.thickness}).")
            
            # 3. Material Existence
            if layer.material not in self.active_material_dict:
                errors.append(f"Layer {i}: Material '{layer.material}' not found in material dictionary.")
                
        return errors

    def get_solver_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pivots the object-oriented structure into columnar NumPy arrays for the solver.
        
        This process involves:
        1.  **Expansion**: Converting logical layers (e.g., inhomogeneous layers) into multiple
            physical sub-layers.
        2.  **Interface Generation**: Inserting interface layers between materials, calculating
            mixed refractive indices (EMA) using Looyenga mixing.
        3.  **Flattening**: Converting the list of objects into Structure-of-Arrays (SoA) layout
            required by `scattering_matrix.py`.

        Returns:
            Tuple containing five strictly typed arrays:
            - indices (complex128): Shape (n_total, n_wavs). Refractive indices per layer/wavelength.
            - thicknesses (float64): Shape (n_total,). Physical thickness in nm.
            - incoherent_flags (bool): Shape (n_total,). True if layer breaks coherence (thick).
            - rough_types (int32): Shape (n_total,). Model type ID for the interface below.
            - rough_vals (float64): Shape (n_total,). Sigma value for the interface below.
        
        Raises:
            ValueError: If the structure layer list is empty.
        """
        if not self.layer_list:
            raise ValueError("Structure is empty.")

        # --- Buffers for Columnar Data ---
        # Using lists for accumulation is generally faster than repeated np.append 
        # or resizing arrays for unknown final sizes (due to inhomogeneity/interfaces).
        col_thick: List[float] = []
        col_nk: List[Union[complex, np.ndarray]] = []
        col_coh: List[bool] = []
        col_r_val: List[float] = []
        col_r_type: List[int] = []

        # --- 1. Ambient (First Layer) ---
        first_layer = self.layer_list[0]
        nk_0 = self.active_material_dict[first_layer.material].nk
        
        col_thick.append(first_layer.thickness)
        col_nk.append(nk_0)
        col_coh.append(first_layer.coherent)
        col_r_val.append(first_layer.roughness)
        col_r_type.append(first_layer.rough_type)

        # --- 2. Iterate Subsequent Layers ---
        # Cache dictionary lookups
        get_group = self.group_dict.get
        get_mat = self.active_material_dict.__getitem__
        
        # We need access to the "previous" effective NK for interface mixing.
        # Initialize prev_eff_nk with ambient
        prev_eff_nk = nk_0 
        
        # Pre-instantiate a default group to avoid checks inside loop
        default_group = Group("default") 

        for i in range(1, len(self.layer_list)):
            layer = self.layer_list[i]
            mat_name = layer.material
            
            # Fast dict access with default fallback
            group = get_group(mat_name, default_group)
            
            # Base NK
            base_nk = get_mat(mat_name).nk
            
            # Apply Group NK Factors
            if group.n_factor != 1.0 or group.k_factor != 1.0:
                layer_nk = base_nk * group.nk_factor
            else:
                layer_nk = base_nk

            # Apply Group Thickness Factors
            layer_thickness = layer.thickness * group.thick_factor + group.thick_summand
            if layer_thickness < 0.0:
                layer_thickness = 0.0

            # --- A. Interface Generation ---
            if layer.interface:
                # Thickness logic
                t_interface = layer.interface_thickness
                layer_thickness -= t_interface
                
                # Mixing Logic (Looyenga)
                # prev_eff_nk is from the iteration i-1. 
                interface_nk = looyenga_eps(layer_nk, prev_eff_nk, 0.5)

                col_thick.append(t_interface)
                col_nk.append(interface_nk)
                col_coh.append(True) # Interfaces are thin/coherent
                col_r_val.append(0.0)
                col_r_type.append(0)

            # --- B. Inhomogeneity Generation ---
            if layer._inhomogen and layer.sub_layer_count > 1:
                sub_div = layer.sub_layer_count
                # Inhomogeneity delta calculation
                total_delta = (layer._inh_delta + group.inh_delta_summand) * 0.5
                
                # Vectorized gradients
                factors = np.linspace(1.0 - total_delta, 1.0 + total_delta, sub_div)
                step_t = layer_thickness / sub_div
                
                # Append steps
                # Note: roughness only applies to the top of the layer (first sub-layer here)
                for ix, f in enumerate(factors):
                    col_thick.append(step_t)
                    col_nk.append(layer_nk * f)
                    col_coh.append(layer.coherent)
                    
                    if ix == 0:
                        col_r_val.append(layer.roughness)
                        col_r_type.append(layer.rough_type)
                    else:
                        col_r_val.append(0.0)
                        col_r_type.append(0)
                        
            # --- C. Standard Layer ---
            else:
                col_thick.append(layer_thickness)
                col_nk.append(layer_nk)
                col_coh.append(layer.coherent)
                col_r_val.append(layer.roughness)
                col_r_type.append(layer.rough_type)

            # Update previous effective NK for next iteration's interface
            prev_eff_nk = layer_nk

        # --- 3. Final Conversion to Solver-Compatible Arrays ---
        
        # Convert lists to NumPy arrays with strict types
        # indices must be (n_layers, n_wavs). vstack handles arrays/scalars correctly.
        out_indices = np.vstack(col_nk).astype(COMPLEX_TYPE)
        out_thick = np.array(col_thick, dtype=FLOAT_TYPE)
        
        # Invert coherence for solver (solver uses incoherent_flags)
        # coherent=True -> incoherent_flag=False
        out_inc_flags = np.array([not c for c in col_coh], dtype=np.bool_)
        
        out_r_types = np.array(col_r_type, dtype=INT_TYPE)
        out_r_vals = np.array(col_r_val, dtype=FLOAT_TYPE)
        
        # --- Populate Legacy List (optional) ---
        self.simple_layer_list = [
            [col_thick[i], col_nk[i], col_coh[i], col_r_val[i], col_r_type[i]]
            for i in range(len(col_thick))
        ]

        return out_indices, out_thick, out_inc_flags, out_r_types, out_r_vals
    
    def get_error_solver_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs the layer stack with stochastic manufacturing errors applied.
        
        Unlike `get_solver_inputs` which uses ideal parameters (or simple optimized factors),
        `tf_architect` activates the `Group` error models based on the `error_mask`. 
        This is used for Yield Analysis or Robustness Optimization.
    
        Error Mask Mapping (from Group):
            0: Thickness Error
            1: n (Real Index) Error
            2: k (Imaginary Index) Error
            3: Roughness Error
            4: Inhomogeneity Delta Error
            5: Interface Thickness Error
    
        Returns:
            Tuple containing the five solver-compatible NumPy arrays (indices, thicknesses, etc.).
        """
        if not self.layer_list:
            raise ValueError("Structure is empty.")
    
        col_thick: List[float] = []
        col_nk: List[Union[complex, np.ndarray]] = []
        col_coh: List[bool] = []
        col_r_val: List[float] = []
        col_r_type: List[int] = []
    
        # --- 1. Ambient (First Layer) ---
        # Generally, ambient is fixed/ideal, but errors *could* be applied if grouped.
        # Assuming ambient usually has no errors for now.
        first_layer = self.layer_list[0]
        nk_0 = self.active_material_dict[first_layer.material].nk
        
        col_thick.append(first_layer.thickness)
        col_nk.append(nk_0)
        col_coh.append(first_layer.coherent)
        col_r_val.append(first_layer.roughness)
        col_r_type.append(first_layer.rough_type)
    
        # --- 2. Iterate Layers ---
        get_group = self.group_dict.get
        get_mat = self.active_material_dict.__getitem__
        prev_eff_nk = nk_0 
        default_group = Group("default")
    
        for i in range(1, len(self.layer_list)):
            layer = self.layer_list[i]
            mat_name = layer.material
            group = get_group(mat_name, default_group)
            
            # 1. Base Values (Systematic)
            base_nk = get_mat(mat_name).nk
            
            # Apply Systematic Group Factors
            if group.n_factor != 1.0 or group.k_factor != 1.0:
                layer_nk = base_nk * group.nk_factor
            else:
                layer_nk = base_nk
    
            layer_thickness = layer.thickness * group.thick_factor + group.thick_summand
    
            # 2. Apply Stochastic Errors (Based on Mask)
            # Mask 0: Thickness
            if group.error_mask[0]:
                layer_thickness = group.thickness_error(layer_thickness)
            
            if layer_thickness < 0.0: 
                layer_thickness = 0.0
    
            # Mask 1 (N) and Mask 2 (K)
            if group.error_mask[1] or group.error_mask[2]:
                n_part = layer_nk.real
                k_part = layer_nk.imag
                
                if group.error_mask[1]:
                    n_part = group._apply_error(n_part, group.n_error_type, group.n_error_params)
                    n_part = np.maximum(0.0, n_part) # Physical constraint
                
                if group.error_mask[2]:
                    k_part = group._apply_error(k_part, group.k_error_type, group.k_error_params)
                
                layer_nk = n_part + 1j * k_part
    
            # Mask 3: Roughness
            current_roughness = layer.roughness
            if group.error_mask[3]:
                current_roughness = group.sr_roughness_error(current_roughness, layer_thickness)
    
            # --- A. Interface Generation ---
            if layer.interface:
                t_interface = layer.interface_thickness
                
                # Mask 5: Interface Error
                if group.error_mask[5]:
                    t_interface = group.interface_error(t_interface, layer.thickness)
    
                # Ensure interface fits in layer
                if t_interface > layer_thickness:
                    t_interface = layer_thickness
    
                layer_thickness -= t_interface
                
                # Looyenga Mixing
                interface_nk = looyenga_eps(layer_nk, prev_eff_nk, 0.5)
    
                col_thick.append(t_interface)
                col_nk.append(interface_nk)
                col_coh.append(True)
                col_r_val.append(0.0)
                col_r_type.append(0)
    
            # --- B. Inhomogeneity Generation ---
            if layer._inhomogen and layer.sub_layer_count > 1:
                sub_div = layer.sub_layer_count
                
                # Systematic Inhomogeneity
                current_delta = (layer._inh_delta + group.inh_delta_summand) * 0.5
    
                # Mask 4: Inhomogeneity Error
                if group.error_mask[4]:
                    current_delta = group.inh_delta_error(current_delta)
                
                factors = np.linspace(1.0 - current_delta, 1.0 + current_delta, sub_div)
                step_t = layer_thickness / sub_div
                
                for ix, f in enumerate(factors):
                    col_thick.append(step_t)
                    col_nk.append(layer_nk * f)
                    col_coh.append(layer.coherent)
                    
                    if ix == 0:
                        col_r_val.append(current_roughness)
                        col_r_type.append(layer.rough_type)
                    else:
                        col_r_val.append(0.0)
                        col_r_type.append(0)
                        
            # --- C. Standard Layer ---
            else:
                col_thick.append(layer_thickness)
                col_nk.append(layer_nk)
                col_coh.append(layer.coherent)
                col_r_val.append(current_roughness)
                col_r_type.append(layer.rough_type)
    
            prev_eff_nk = layer_nk
    
        # --- 3. Final Conversion ---
        out_indices = np.vstack(col_nk).astype(COMPLEX_TYPE)
        out_thick = np.array(col_thick, dtype=FLOAT_TYPE)
        out_inc_flags = np.array([not c for c in col_coh], dtype=np.bool_)
        out_r_types = np.array(col_r_type, dtype=INT_TYPE)
        out_r_vals = np.array(col_r_val, dtype=FLOAT_TYPE)
    
        return out_indices, out_thick, out_inc_flags, out_r_types, out_r_vals