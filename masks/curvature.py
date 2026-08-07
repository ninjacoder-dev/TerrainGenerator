"""
masks/curvature.py — Carte de courbure.

Distingue les zones convexes (arêtes) et concaves (creux).
Très utile pour l'accumulation de neige, mousse, eau.
"""

import numpy as np
from utils.math_utils import curvature_map, normalize


def generate_curvature_mask(heightmap: np.ndarray) -> np.ndarray:
    """Generate normalized mean curvature map."""
    curv = curvature_map(heightmap)
    return normalize(curv)
