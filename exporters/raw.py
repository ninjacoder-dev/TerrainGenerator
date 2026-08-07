"""
exporters/raw.py — Export format RAW (Unreal Engine / Unity uint16 binary format).
"""

import numpy as np
from pathlib import Path


def export_raw(array: np.ndarray, filepath: str) -> str:
    """
    Save 16-bit raw binary heightmap file (.r16 / .raw).
    Format used by Unreal Engine and Unity.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    raw_data = (np.clip(array, 0.0, 1.0) * 65535.0).astype('<u2')

    with open(filepath, 'wb') as f:
        raw_data.tofile(f)

    print(f"  [export] Saved RAW 16-bit binary: {filepath}")
    return filepath
