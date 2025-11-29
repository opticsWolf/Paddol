# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

The core design of this data container favors **speed** over memory consumption, 
and relies on Python's native high-performance tools:

* **Measurement Data:** Stored in flat `numpy.ndarray` objects using the 
    `float32` dtype to minimize memory footprint for large spectral arrays.
* **Data Structure:** A nested Python `dict` provides O(1) average-time 
    lookup for the three-level key (angle, polarization, spectral type). 
    The overhead of the nesting is negligible compared to the cost of
    managing the large underlying arrays.
* **Wavelengths:** Associated wavelength arrays are stored per-angle, 
    with robust checks to ensure data consistency across all measurements 
    taken at the same angle. A cached union of all unique wavelengths is 
    lazily computed for efficient external use.
* **Efficiency:** Key mapping methods (`__getitem__`, `__setitem__`, `__contains__`) 
    operate in O(1) average time, and the container's length (`__len__`) is 
    maintained via a running count, also ensuring **O(1) time complexity**.
"""

import numpy as np
from typing import Tuple, Dict, List, Union, Iterable, Any, Optional


# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 11:31:13 2025

@author: Frank
"""
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


class NestedOpticalData:
    """
    Dictionary‑like container for optical data indexed by a three‑level key.

    The key is a 3‑tuple:

        (angle_degrees: float,
         polarization: str,
         spectral_type: str)

    Internally the mapping is::

        self._data[angle][polarization][spectral_type] = np.ndarray

    All leaf arrays are stored as ``np.float32`` to keep memory usage low.
    Wavelengths are kept per‑angle in a separate dictionary.  A cached union
    of all wavelengths is lazily recomputed only when needed.

    Attributes
    ----------
    _data : dict[float, dict[str, dict[str, np.ndarray]]]
        Nested mapping from angle → polarization → spectral_type.
    _wavelengths : dict[float, np.ndarray]
        Per‑angle wavelength arrays.
    _cached_all_wavelengths : Optional[np.ndarray]
        Cached union of all wavelengths; ``None`` means “needs recompute”.
    _length_count : int
        Running count of the total number of unique (angle, polarization, spectral_type) keys.
    """

    def __init__(self) -> None:
        self._data: dict[float, dict[str, dict[str, np.ndarray]]] = {}
        self._wavelengths: dict[float, np.ndarray] = {}
        self._cached_all_wavelengths: Optional[np.ndarray] = None
        self._length_count: int = 0

    # ------------------------------------------------------------------ #
    # Core helpers
    # ------------------------------------------------------------------ #

    def _ensure_angle(self, angle: float) -> None:
        """Create nested dictionaries for a new *angle* if needed."""
        if angle not in self._data:
            self._data[angle] = {}

    # ------------------------------------------------------------------ #
    # Mapping interface
    # ------------------------------------------------------------------ #

    def __getitem__(self, key: Tuple[float, str, str]) -> np.ndarray:
        """Return the array stored at *key*.

        Raises
        ------
        KeyError
            If any part of the key hierarchy is missing.
        """
        angle, pol, spec = key
        return self._data[angle][pol][spec]

    def __setitem__(
        self,
        key: Tuple[float, str, str],
        value: Union[np.ndarray, Iterable[Any]],
        wavelength: Optional[Union[np.ndarray, Iterable[Any]]] = None,
    ) -> None:
        """
        Store *value* under *key*, optionally updating the wavelengths.

        Parameters
        ----------
        key : tuple
            (angle_degrees, polarization, spectral_type)
        value : array‑like
            Data to store.  Converted to ``np.ndarray`` of dtype ``float32``.
        wavelength : array‑like or None
            Optional wavelength array that must match *value*'s length.
            If omitted and the angle is new, data can be stored first; a later call
            to :py:meth:`set_wavelengths` will attach wavelengths.  If the angle
            already exists and wavelengths are present, the supplied wavelength
            **must** match the existing one.

        Raises
        ------
        ValueError
            When ``wavelength`` is provided but its length does not match *value*,
            or when it conflicts with an existing wavelength for that angle.
        """
        angle, pol, spec = key

        # Convert data to np.ndarray first – needed for the length check
        arr = np.asarray(value, dtype=np.float32)

        if wavelength is None:
            # Data can be added without wavelengths (for a new or already‑existing angle).
            pass
        else:
            wl_arr = np.asarray(wavelength, dtype=np.float32)
            if len(wl_arr) != len(arr):
                raise ValueError(
                    f"Wavelength length ({len(wl_arr)}) does not match data length "
                    f"({len(arr)})."
                )
            # If wavelengths already exist for this angle, ensure they match.
            existing_wl = self._wavelengths.get(angle)
            if existing_wl is not None and not np.array_equal(existing_wl, wl_arr):
                raise ValueError(
                    "Provided wavelength array conflicts with the existing "
                    f"wavelengths for angle {angle}."
                )
            # Store/overwrite wavelengths for this angle.
            self._wavelengths[angle] = wl_arr
            # Invalidate the cached union – a new or changed wavelength may have appeared.
            self._cached_all_wavelengths = None

        # Persist the data
        self._ensure_angle(angle)
        self._data[angle].setdefault(pol, {})[spec] = arr

        #Determine if the key is new (True=1, False=0) and add the result to the counter.
        self._length_count += key not in self

    def __contains__(self, key: Tuple[float, str, str]) -> bool:
        """Return ``True`` iff *key* exists."""
        try:
            _ = self[key]
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        """Return the total number of stored 3‑tuple keys."""
        return self._length_count

    def keys(self) -> Iterable[Tuple[float, str, str]]:
        """
        Iterate over all stored 3‑tuples.

        Yields
        ------
        tuple
            (angle, polarization, spectral_type)
        """
        for angle, pol_dict in self._data.items():
            for pol, spec_dict in pol_dict.items():
                for spec in spec_dict:
                    yield angle, pol, spec

    def values(self) -> Iterable[np.ndarray]:
        """
        Iterate over all stored numpy arrays.

        Yields
        ------
        numpy.ndarray
            The measurement data array.
        """
        for pol_dict in self._data.values():
            for spec_dict in pol_dict.values():
                for value in spec_dict.values():
                    yield value

    def items(self) -> Iterable[Tuple[Tuple[float, str, str], np.ndarray]]:
        """
        Iterate over all stored (key, value) pairs.

        Yields
        ------
        tuple
            ((angle, polarization, spectral_type), numpy.ndarray)
        """
        for angle, pol_dict in self._data.items():
            for pol, spec_dict in pol_dict.items():
                for spec, value in spec_dict.items():
                    yield (angle, pol, spec), value

    # ------------------------------------------------------------------ #
    # Convenience methods
    # ------------------------------------------------------------------ #

    def get_as_percent(self, key: Tuple[float, str, str]) -> float:
        """
        Return the mean of *key* multiplied by 100.

        For `'T'`, `'R'` and `'A'` spectra this is a common way of reporting
        values as percentages.

        Returns
        -------
        float
            Mean percent value.
        """
        _, _, spec = key
        if spec not in ("T", "R", "A"):
            raise ValueError(
                "Spectral type must be 'T', 'R' or 'A' for percent conversion"
            )
        return self[key] * 100.0

    def get_as_degrees(self, key: Tuple[float, str, str]) -> np.ndarray:
        """
        Return the data in degrees.

        For `'Psi'` and `'Delta'` spectra this converts radians to degrees.

        Returns
        -------
        numpy.ndarray
            Same shape as ``self[key]``.
        """
        _, _, spec = key
        if spec not in ("Psi", "Delta"):
            raise ValueError(
                "Spectral type must be 'Psi' or 'Delta' for degree conversion"
            )
        return np.degrees(self[key])

    # ------------------------------------------------------------------ #
    # Wavelength handling
    # ------------------------------------------------------------------ #

    def set_wavelengths(
        self,
        angle_deg: float,
        wavelengths: Union[np.ndarray, Iterable[Any]],
    ) -> None:
        """
        Store a wavelength array for a specific angle.

        Parameters
        ----------
        angle_deg : float
            The incidence angle.
        wavelengths : array‑like
            Must be one‑dimensional and match the length of all data stored
            under *angle_deg* (the check is performed lazily when new data are
            added).  If data already exist for this angle, the new wavelength
            must have the same length as every existing array.

        Raises
        ------
        ValueError
            If lengths do not match existing data.
        """
        wl_arr = np.asarray(wavelengths, dtype=np.float32)

        # Validate against existing data (if any)
        if angle_deg in self._data:
            for pol_dict in self._data[angle_deg].values():
                for spec_arr in pol_dict.values():
                    if len(spec_arr) != len(wl_arr):
                        raise ValueError(
                            f"Wavelength length ({len(wl_arr)}) does not match "
                            "existing data length for angle {angle_deg}."
                        )

        self._wavelengths[angle_deg] = wl_arr
        self._cached_all_wavelengths = None

    def get_wavelengths(
        self,
        angle_deg: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return wavelengths for a specific angle or all unique wavelengths.

        Parameters
        ----------
        angle_deg : float, optional
            If supplied, the wavelength array stored under that angle is returned.
            Otherwise a combined array of *all* wavelengths is produced and cached.

        Returns
        -------
        numpy.ndarray
            The requested wavelength data (float32).  Raises ``KeyError`` if an
            angle has no wavelengths set yet.
        """
        if angle_deg is not None:
            try:
                return self._wavelengths[angle_deg]
            except KeyError as exc:
                raise KeyError(
                    f"No wavelengths defined for angle {angle_deg}."
                ) from exc

        # Return the cached union – recompute only when necessary.
        if self._cached_all_wavelengths is None:
            self._cached_all_wavelengths = self._combine_all_wavelengths()
        return self._cached_all_wavelengths

    def _combine_all_wavelengths(self) -> np.ndarray:
        """Internal helper that returns the unique union of all wavelength arrays."""
        if not self._wavelengths:
            return np.array([], dtype=np.float32)

        concatenated = np.concatenate(list(self._wavelengths.values()))
        return np.unique(concatenated, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def mean(self, key: Tuple[float, str, str]) -> float:
        """Return the arithmetic mean of ``self[key]``."""
        return float(np.mean(self[key], dtype=np.float32))

    def median(self, key: Tuple[float, str, str]) -> float:
        """Return the median (50th percentile) of ``self[key]``."""
        return float(np.median(self[key]))

    def std_dev(self, key: Tuple[float, str, str]) -> float:
        """Return the standard deviation of ``self[key]``."""
        return float(np.std(self[key], dtype=np.float32))

    def variance(self, key: Tuple[float, str, str]) -> float:
        """Return the variance of ``self[key]``."""
        # Note: In NumPy, np.var() typically uses a default 'ddof=0' (population variance).
        return float(np.var(self[key], dtype=np.float32))

    def min_val(self, key: Tuple[float, str, str]) -> float:
        """Return the minimum value of ``self[key]``."""
        return float(np.min(self[key]))

    def max_val(self, key: Tuple[float, str, str]) -> float:
        """Return the maximum value of ``self[key]``."""
        return float(np.max(self[key]))

    # ------------------------------------------------------------------ #
    # Equality & representation
    # ------------------------------------------------------------------ #

    def __eq__(self, other: Any) -> bool:
        """
        Compare two :class:`NestedOpticalData` objects for equality.

        Parameters
        ----------
        other : NestedOpticalData or dict
            Another instance of this class or a plain dictionary that follows
            the same key → array mapping contract.

        Returns
        -------
        bool
            ``True`` iff all data and wavelength mappings are equal.
        """
        if isinstance(other, NestedOpticalData):
            # Compare measurement data --------------------------------------
            for k in self.keys():
                if k not in other:
                    return False
                if not np.array_equal(self[k], other[k]):
                    return False

            # Compare wavelength data ---------------------------------------
            if set(self._wavelengths) != set(other._wavelengths):
                return False
            for angle, wl_self in self._wavelengths.items():
                wl_other = other._wavelengths.get(angle)
                if not np.array_equal(wl_self, wl_other):
                    return False

            # Compare cached union (optional) ---------------------------------
            return np.array_equal(
                self.get_wavelengths(), other.get_wavelengths()
            )

        elif isinstance(other, dict):  # plain dict mapping key→array
            for k in self.keys():
                if k not in other or not np.array_equal(self[k], other[k]):
                    return False
            return True

        return NotImplemented

    def __repr__(self) -> str:
        """Return a concise representation."""
        n_keys = len(list(self.keys()))
        # Grab the shape of one array (if any) for illustration.
        sample_shape: Tuple[int, ...] | None = None
        try:
            first_key = next(iter(self.keys()))
            sample_shape = self[first_key].shape
        except StopIteration:
            pass

        shape_str = f", sample shape={sample_shape}" if sample_shape else ""
        return f"<NestedOpticalData with {n_keys} keys{shape_str}>"

    # ------------------------------------------------------------------ #
    # Partial key lookup
    # ------------------------------------------------------------------ #

    def get_by_partial_key(
        self,
        partial_key: Tuple[Optional[float], Optional[str], Optional[str]],
    ) -> Dict[Tuple[float, str, str], np.ndarray]:
        """
        Return a dictionary of all keys and values that match the given partial key.

        Parameters
        ----------
        partial_key : tuple
            A 3‑tuple where each element can be ``None`` to act as a wildcard.
            Example: ``(45.0, None, 'T')`` returns all transmittance data at
            45° regardless of polarization.

        Returns
        -------
        dict
            Keys are full 3‑tuples and values are the corresponding numpy arrays.
        """
        result: Dict[Tuple[float, str, str], np.ndarray] = {}
        for angle, pol_dict in self._data.items():
            if partial_key[0] is not None and angle != partial_key[0]:
                continue
            for pol, spec_dict in pol_dict.items():
                if partial_key[1] is not None and pol != partial_key[1]:
                    continue
                for spec, value in spec_dict.items():
                    if partial_key[2] is not None and spec != partial_key[2]:
                        continue
                    result[(angle, pol, spec)] = value
        return result