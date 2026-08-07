"""
biome/vegetation.py — Placement de la végétation globale.
"""

import numpy as np
from utils.math_utils import clamp


def generate_vegetation_mask(grass_mask: np.ndarray,
                             forest_mask: np.ndarray) -> np.ndarray:
    """Combine grass and forest masks into overall vegetation index."""
    return clamp(grass_mask * 0.5 + forest_mask * 0.5, 0.0, 1.0)
