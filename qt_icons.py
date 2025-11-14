# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

from pathlib import Path
from typing import Dict, Union

def scan_icons_folder(folder_path: Union[str, Path]) -> Dict[str, str]:
    """Scan *folder_path* for icon files and return a mapping of base names to paths.

    The function looks only at regular files whose extensions are ``.png`` or
    ``.svg`` (case‑insensitive).  For each matching file the key in the returned
    dictionary is the filename without its extension (*stem*), while the value
    is the absolute path as a string, which can be supplied directly to PyQt
    widgets.

    Args:
        folder_path: Directory containing icon assets. Can be a ``str`` or
            :class:`pathlib.Path`.

    Returns:
        dict[str, str]: Mapping from icon base name to absolute file path.
    """
    folder = Path(folder_path)
    icon_dict: Dict[str, str] = {}
    if not folder.is_dir():
        return icon_dict
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in {".png", ".svg"}:
            icon_dict[file.stem] = str(file.resolve())
    return icon_dict

try:
    ICON_DICT = scan_icons_folder(Path("src/icons"))
except Exception as e:
    print(f"Error initializing icons: {e}")