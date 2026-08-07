"""
masks/slope.py — Carte de pente.

Calcul : gradient(heightmap)
Utilisé pour placer falaises, herbe, neige, rochers.
"""

import numpy as np
from utils.math_utils import slope_map, normalize


def generate_slope_mask(heightmap: np.ndarray) -> np.ndarray:
    """Generate normalized slope map from heightmap."""
    slopes = slope_map(heightmap)
    return normalize(slopes)
