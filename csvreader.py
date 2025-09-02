# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 23:21:32 2025

PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import csv
import re
import io
import chardet
import numpy as np
import polars as pl
from typing import List
from itertools import islice

class CSVReader:
    def __init__(self, sep=None, decimal=None):
        """
        Utility for reading CSV files with irregular headers.

        The constructor accepts optional separator and decimal marker arguments.
        If they are omitted, the underlying Polars/CSV reader will attempt to
        auto‑detect them.  The instance also keeps a set of keyword lists that
        help identify wavelength, spectral, and cleaning columns in a dataset.

        Args:
            sep (str | None, optional):
                Field separator.  If ``None`` the format is auto‑detected.
            decimal (str | None, optional):
                Decimal marker (either '.' or ',').  If ``None`` it will be
                detected automatically.

        Attributes:
            encoding (str | None):
                File encoding used when reading text.  It defaults to ``None``
                (Polars/CSV reader will use its own auto‑detection).  You can set
                it manually if you know the exact encoding of your files.
            raw_text (str | None):
                Holds the raw CSV content after a read operation; useful for
                debugging or re‑parsing without reopening the file.
            wavl_keywords (set[str]):
                Keywords that identify wavelength columns.
            spectral_keywords (set[str]):
                Keywords that identify spectral data columns.
            remove_keywords (set[str]):
                Substrings to strip from column names during cleaning.
            special_chars (set[str]):
                Characters to delete from column names during cleaning.

        Notes:
            * The keyword sets are intentionally small and case‑insensitive
              because the typical CSV files in this domain use short,
              lower‑case identifiers.  If a larger set is required, subclass
              ``CSVReader`` and override the relevant attributes.
            * ``raw_text`` is not populated until a file is read; attempting to
              access it before that will raise an ``AttributeError``.
        """
        self.sep = sep
        self.decimal = decimal
        self.encoding = None
        
        self.raw_text = None
        
        # keyword sets
        self.wavl_keywords = {"wavl", "wavelength", "wavls", "wvls", "wvl", "wavelengths"}
        self.spectral_keywords = {"r", "t", "a", "reflection", "transmission", "absorption"}
        self.remove_keywords = {"normalize", "absolute", "data"}
        self.special_chars = {"#", "(", ")", "[", "]"}

    def detect_csv_format(self, filepath: str, sample_size: int = 5000) -> dict:
        """
        Detect separator, decimal and encoding from a small file fragment.

        Args
            filepath : str
                Path to the file.
            sample_size : int, optional
                Number of bytes to read for detection. Defaults to 5000.

        Returns
            dict
                ``{'separator': sep, 'decimal': dec, 'encoding': enc}``
        """
        # --- encoding ----------------------------------------------------
        with open(filepath, "rb") as f:
            raw = f.read(sample_size)
        self.encoding = chardet.detect(raw)["encoding"] or "utf-8"

        # --- sample & separator -----------------------------------------
        with open(filepath, "r", encoding=self.encoding, errors="ignore") as f:
            sample = f.read(sample_size)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
            self.sep = dialect.delimiter
        except csv.Error as exc:          # fallback to comma if sniffing fails
            print(f"Sniffer error: {exc}")
            self.sep = ","

        # --- decimal symbol ------------------------------------------------
        if self.sep == ",":
            self.decimal = "."
        else:
            self.decimal = "," if sample.count(",") > sample.count(".") and re.search(r"\d+,\d+", sample) else "."

        return {"separator": self.sep, "decimal": self.decimal, "encoding": self.encoding}
    
    def _clean_name(self, name: str) -> str:
        """
        Normalise a column name.
        
        Removes user‑defined keywords and special characters,
        collapses whitespace to underscores, and strips leading/trailing
        underscores.
        """
        if not name:
            return ""
        for bad in self.remove_keywords:
            name = re.sub(bad, "", name, flags=re.I)
        for ch in self.special_chars:
            name = name.replace(ch, "")
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name

    def _is_wavl_col(self, col_name: str) -> bool:
        """
        Return ``True`` if *col_name* contains any wavelength keyword.
        """
        return any(kw in col_name.lower() for kw in self.wavl_keywords)
    
    def _read_raw_csv(self, path):
        if self.encoding is None:
            self.detect_csv_format(path)
        with open(path, mode="r", newline="", encoding=self.encoding) as f:
            self.raw_text = f.read()
    
    def read_csv(self, path: str,
            header_skip: int = 0,
            header_start: int = 0,
            header_stop: int = 1,
            auto_remove_comments:bool = True
            ) -> pl.DataFrame:
        """
        Read a CSV file with an uncertain header structure.
        1. The raw text (for display/debugging)
        2. A list of parsed rows with the first ``header_skip`` rows omitted.
        The function first parses the file to locate a row that contains any of
        ``self.wavl_keywords`` (e.g. “380”).  It then constructs column names from
        a small window around that row, ensures those names are unique, and finally
        reads the data with Polars.

        Args:
            path (str): Path to the CSV file.
    
        Returns:
            pl.DataFrame: DataFrame with unique column names and a single
                wavelength column named ``"wavl"``.
    
        Raises:
            ValueError: If no wavelength header row or column is found.
            DuplicateError: If duplicate column names are detected after
                deduplication (should not occur).
                
        Notes:
            * All keyword sets are public and can be modified after instantiation if a
              user needs to extend or shrink the search space.  
        """
            
        # 1. Load the raw file once – this is very fast for small to medium CSVs
        if self.encoding is None:
            self.detect_csv_format(path)
        if self.raw_text is None:
            with open(path, mode="r", newline="", encoding=self.encoding) as f:
                self.raw_text = f.read()
        reader = csv.reader(io.StringIO(self.raw_text))
        rows: List[List[str]] = list(islice(reader, header_skip, None))

        # 2. Remove comment lines & empty lines -------------------------------
        if auto_remove_comments:
            rows = [
                r for r in rows
                if r and not r[0].lstrip().startswith("#")          # not a comment
                and any(cell.strip() for cell in r)                # not empty
            ]
        
            if not rows:                     # file became empty after filtering
                raise ValueError(
                    "File contains only comments or is otherwise empty"
                )

        # 3. Find the row that contains a wavelength keyword ------------------
        base_idx: int | None = None
        for i, row in enumerate(rows):
            if any(
                kw in c.lower().strip()
                for kw in self.wavl_keywords
                for c in row
            ):
                base_idx = i
                break

        if base_idx is None:
            raise ValueError("No wavelength header row found!")

        # 4. Grab a few rows around the header to build names -----------------
        header_idxs: List[int] = [
            j for j in range(base_idx - header_start, base_idx + header_stop) if 0 <= j < len(rows)
        ]
        header_rows = [rows[j] for j in header_idxs]

        ncols = len(rows[base_idx])
        col_names: List[str] = []

        # 5. Build names and make them unique ---------------------------------
        used_names: set[str] = set()
        for j in range(ncols):
            parts: List[str] = []
            for hr in header_rows:
                if j < len(hr) and hr[j].strip():
                    parts.append(hr[j].strip())
            col_clean = "_".join(parts) if parts else f"col{j}"
            col_clean = self._clean_name(col_clean)

            # Ensure uniqueness by appending an incrementing suffix
            original = col_clean
            counter = 1
            while col_clean in used_names:
                col_clean = f"{original}_{counter}"
                counter += 1

            used_names.add(col_clean)
            col_names.append(col_clean)

        # 6. Detect format if necessary ---------------------------------------
        if self.sep is None or self.decimal is None:
            self.detect_csv_format(path)  # implementation omitted for brevity

        skip_rows = max(header_idxs) + 1
        
        csv_text = "\n".join(
            self.sep.join(cell.strip() for cell in row)
            for row in rows[skip_rows:])
        csv_file_like = io.StringIO(csv_text)
        
        # 7. Read the data with Polars and assign unique column names ---------
        df = pl.read_csv(
            csv_file_like,
            separator=self.sep,
            decimal_comma=(self.decimal == ","),
            has_header=False,
            skip_rows=0,
        )
        
        df = df[[s.name for s in df if not (s.null_count() == df.height)]]  # drop a row if all values are all null
        df = df.filter(~pl.all_horizontal(pl.all().is_null()))  # drop a row if all values are all null
        #df = df.with_columns([pl.col("*").cast(pl.Float64)])
        df.columns = col_names
        # 8. Find the wavelength column(s) --------
        wavl_cols = [c for c in df.columns if any(kw in c.lower() for kw in self.wavl_keywords)]
        if not wavl_cols:
            raise ValueError("No wavelength column found!")
        return df, self.raw_text

    def auto_map_headers(self, df: pl.DataFrame):
        """Return dict of structured metadata and a renamed dataframe."""
        meta_map = {}
        rename_map = {}

        for col in df.columns:
            if col == "wavl":
                continue
            else:
                #print ('col', col)
                meas, pol, aoi = self._map_column(col)
                #print (meas, pol, aoi)
                canonical = f"{meas}_{pol}" if meas else col
                if aoi is not None:
                    canonical = f"{canonical}_{aoi:g}"
                rename_map[col] = canonical
                #print (rename_map)
                meta_map[col] = {"measurement": meas, "polarization": pol, "aoi": aoi}

        df = df.rename(rename_map)
        #self.col_mapping.update(meta_map)
        return df, meta_map

    def _map_column(self, col: str):
        """Map a single cleaned column name to (measurement, polarization, AOI)."""
        name = col.lower()

        meas, pol = None, "a"  # default pol = average
        # --- Measurement type ---
        if re.search(r"(reflect|^r$|^[r]|[^a-z]r[^a-z]?|ra|rs|rp)", name):
            meas = "R"
        elif re.search(r"(transmit|^t$|^[t]|[^a-z]t[^a-z]?|ta|ts|tp)", name):
            meas = "T"
        elif re.search(r"(absor|^a$|^[a]|[^a-z]a[^a-z]?|aa|as|ap)", name):
            meas = "A"

        # --- Polarization ---
        if re.search(r"(rs|as|ts|[^a-z]s[^a-z]|s[^a-z]|\bs\b|s-pol|s_polarization)", name):
            pol = "s"
        elif re.search(r"(rp|ap|tp|[^a-z]p[^a-z]|p[^a-z]|\bp\b|p-pol|p_polarization)", name):
            pol = "p"
        elif re.search(r"(ra|aa|ta)", name):
            pol = "a"

        # --- AOI detection ---
        numbers = re.findall(r"\d+(?:\.\d+)?", name)
        aoi = None
        if numbers:
            nums = sorted(float(x) for x in numbers)
            aoi = nums[0]  # use lowest as AOI

        return meas, pol, aoi

    def group_and_align_blocks(
        self,
        df: pl.DataFrame,
        method: str = "interp",
                ) -> dict[str, pl.DataFrame]:
        """
        Group spectral data by wavelength blocks and align them.
    
        The function first extracts every block that starts with a wavelength column.
        All wavelengths from all blocks are merged into one sorted, unique list
        (`global_wavl`).  Every spectral column in every block is then interpolated
        onto this global grid (or dropped if ``method`` == "drop").  Finally,
        any residual `NaN` values are removed so that each returned DataFrame has the
        same number of rows.
    
        Args:
            df (pl.DataFrame):
                DataFrame that contains one or more columns whose names
                match any of ``self.wavl_keywords``.
            method (str, default="interp"):
                Alignment strategy for missing values.
                * ``"drop"`` – drop rows with NaN in a spectral column.
                * ``"interp"`` – linearly interpolate missing values.
    
        Returns:
            dict[str, pl.DataFrame]:
                Mapping from the original wavelength‑column name to an
                aligned DataFrame.  Each returned DataFrame contains a
                column named ``"wavl"`` followed by all spectral columns
                belonging to that block.
    
        Raises:
            ValueError: If no wavelength columns are found or if an
                unsupported ``method`` is supplied.
        """
        
        # 1. Detect all wavelength columns in the original order --------------
        wavl_cols = [c for c in df.columns if self._is_wavl_col(c)]
        if not wavl_cols:
            raise ValueError("No wavelength columns found!")

        col_order = list(df.columns)
        subdfs: list[pl.DataFrame] = []

        blocks: dict[str, pl.DataFrame] = {}
        col_order = list(df.columns)

        # 2. Build each block and keep a reference to the raw DataFrames ------
        for i, wavl_col in enumerate(wavl_cols):
            #print (i, wavl_col)
            start_idx = col_order.index(wavl_col)
            end_idx = (
                col_order.index(wavl_cols[i + 1]) if i + 1 < len(wavl_cols) else len(col_order)
            )
            block_cols = col_order[start_idx:end_idx]
            
            subdf = (
                df.select(block_cols)
                  .rename({wavl_col: "wavl"})
                  .sort("wavl")
                  .unique(subset="wavl")          # keep first occurrence
            )
            subdfs.append(subdf)
           # print (subdf)

        # 3. Build the global wavelength grid (sorted, unique values) ---------
        global_wavl = np.unique(np.concatenate([sub["wavl"].to_numpy() for sub in subdfs]))
        global_wavl.sort()
        global_wavl = global_wavl[~np.isnan(global_wavl)]

        blocks: dict[str, pl.DataFrame] = {}
        
        # 4. Interpolate / drop each block onto the global grid.
        for i, wavl_col in enumerate(wavl_cols):
            subdf = subdfs[i]
            aligned: dict[str, np.ndarray] = {"wavl": global_wavl}
    
    
            if method == 'interp':
                for col in subdf.columns:
                    if col == "wavl":
                        continue
        
                    y_orig = subdf[col].to_numpy()
                    wavl_orig = subdf["wavl"].to_numpy()
                    mask = ~np.isnan(y_orig)
                    if np.sum(mask) < 2:
                        # Not enough points to interpolate → use NaNs everywhere
                        y_aligned = np.full(global_wavl.shape, np.nan)
                    else:
                        y_aligned = np.interp(
                            global_wavl,
                            wavl_orig[mask],
                            y_orig[mask],
                        )
                    aligned[col] = y_aligned
                    
                block_df = pl.DataFrame(aligned)
                blocks[wavl_col] = block_df

            elif method == 'drop':
                subdf.drop_nans()
                for col in subdf.columns:
                    if col == "wavl":
                        continue
                    aligned[col] = subdf[col]
    
                block_df = pl.DataFrame(aligned).drop_nulls()
                blocks[wavl_col] = block_df
    
            elif method == 'trim':
                for col in subdf.columns:
                    if col == "wavl":
                        continue
                    aligned[col] = subdf[col]
                    
                block_df = pl.DataFrame(aligned)
                blocks[wavl_col] = block_df
                
            else:
                raise ValueError("method must be 'drop' or 'interp'")
    
    
        # 6. Global trimming – remove rows that contain a NaN in *any* block --
        if method == "trim":
            # Build a mask that is True only for rows where every block has no NaNs.
            mask_global = np.ones(global_wavl.shape[0], dtype=bool)
            for blk in blocks.values():
                mask_global &= ~np.isnan(blk.drop("wavl").to_numpy()).any(axis=1)
        
            # Apply the global mask to all blocks
            for key, blk in blocks.items():
                blocks[key] = blk.filter(mask_global)
    
        return blocks



# # --------------------------------------------------------------------------- #
# # Demo using an in‑memory file-like object (io.StringIO)
# # --------------------------------------------------------------------------- #

# from pathlib import Path

# demo_csv = """\
# # Sample spectroscopic data
# # Wavelengths are in nm, reflectance (%) with AOI and polarisation markers

# wavls,Rs30 , p_R_30 , R_a_30 , R_s_45 , R_p_45 , R_a_45
# 380,0.13 , 0.16 , 0.11 , 0.14 , 0.17 , 0.12
# 381,0.12 , 0.15 , 0.10 , 0.13 , 0.16 , 0.11
# 382,0.11 , 0.14 , 0.09 , 0.12 , 0.15 , 0.10
# 383,0.13 , 0.14 , 0.09 , 0.12 , 0.15 , 0.10
# 384,0.14 , 0.15 , 0.10 , 0.13 , 0.16 , 0.11
# 401,0.80 , 0.82 , 0.78 , 0.81 , 0.83 , 0.79
# 402,0.79 , 0.81 , 0.77 , 0.80 , 0.82 , 0.78
# 403,0.81 , 0.83 , 0.79 , 0.82 , 0.84 , 0.80
# 404,0.82 , 0.84 , 0.80 , 0.83 , 0.85 , 0.81
# 421,0.05 , 0.04 , 0.06 , 0.05 , 0.04 , 0.06
# 422,0.04 , 0.03 , 0.05 , 0.04 , 0.03 , 0.05
# 423,0.06 , 0.05 , 0.07 , 0.06 , 0.05 , 0.07
# 424,0.07 , 0.06 , 0.08 , 0.07 , 0.06 , 0.08
# 425,0.08 , 0.07 , 0.09 , 0.08 , 0.07 , 0.09
# 426,0.08 , 0.07 , 0.09 , 0.08 , 0.07 , 0.09
# """

# # Write demo to a temporary file (Polars requires a real path)
# tmp_path = Path("demo_spectra.csv")
# tmp_path.write_text(demo_csv, encoding="utf-8")

# # --------------------------------------------------------------------------- #
# # Run the pipeline
# # --------------------------------------------------------------------------- #

# reader = CSVReader()

# # Detect format automatically
# fmt_info = reader.detect_csv_format(str(tmp_path))
# print("\nDetected format:", fmt_info)

# # Read raw data – note that Polars will use the detected separator and decimal
# raw_df, raw_csv = reader.read_csv(str(tmp_path), header_skip=0, header_start=1, header_stop=1, auto_remove_comments=True )
# print("\nRaw DataFrame:")
# print(raw_df.head())

# print("\nRaw CSV:")
# print(raw_csv)

# # Normalise column names
# norm_df, _ = reader.auto_map_headers(raw_df)
# print("\nNormalised DataFrame:")
# print(norm_df.head())

# #Group & align spectral blocks (interpolated onto a common wavelength grid)
# aligned_blocks = reader.group_and_align_blocks(norm_df, method="interp")

# print("\nAligned blocks:")
# for key, block in aligned_blocks.items():
#     print(f"\nBlock originating from column: {key}")
#     print(block.head())
#     print(key)
#     print(block)

# # --------------------------------------------------------------------------- #
# # Clean up temporary file
# # --------------------------------------------------------------------------- #

# tmp_path.unlink()

