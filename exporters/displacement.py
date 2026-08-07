"""
exporters/displacement.py — Export Displacement Map.
"""

from .png16 import export_png16
import numpy as np


def export_displacement(heightmap: np.ndarray, filepath: str) -> str:
    """Export displacement heightmap."""
    return export_png16(heightmap, filepath)
