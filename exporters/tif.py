"""
exporters/tif.py — Export TIFF 16-bit / 32-bit.
"""

import numpy as np
import tifffile
from pathlib import Path


def export_tif(array: np.ndarray, filepath: str, bit_depth: int = 16) -> str:
    """Save array as 16-bit or 32-bit TIFF image."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if bit_depth == 32:
        data = array.astype(np.float32)
    else:
        data = (np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)

    tifffile.imwrite(filepath, data)
    print(f"  [export] Saved TIFF {bit_depth}-bit: {filepath}")
    return filepath
