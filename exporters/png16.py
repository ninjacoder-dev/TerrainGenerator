"""
exporters/png16.py — Export PNG 16-bit.
"""

from PIL import Image
import numpy as np
from pathlib import Path


def export_png16(array: np.ndarray, filepath: str) -> str:
    """
    Save 2D float array (0.0 - 1.0) as 16-bit grayscale PNG.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Scale to 16-bit int (0 - 65535)
    array_clamped = np.clip(array, 0.0, 1.0)
    uint16_data = (array_clamped * 65535.0).astype(np.uint16)

    img = Image.fromarray(uint16_data, mode='I;16')
    img.save(filepath)
    print(f"  [export] Saved 16-bit PNG: {filepath}")
    return filepath
