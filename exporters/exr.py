"""
exporters/exr.py — Export EXR Float32 heightmaps.
"""

import numpy as np
from pathlib import Path
from PIL import Image


def export_exr(array: np.ndarray, filepath: str) -> str:
    """
    Export float32 heightmap to EXR or float TIFF/PNG fallback.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio
        imageio.imwrite(filepath, array.astype(np.float32))
    except Exception:
        # Fallback to float TIFF
        import tifffile
        tifffile.imwrite(filepath, array.astype(np.float32))

    print(f"  [export] Saved float heightmap: {filepath}")
    return filepath
