"""
erosion/sediment.py — Gestion et carte des sédiments.

Gère l'accumulation et le dépôt des sédiments issus de l'érosion.
"""

import numpy as np
from utils.math_utils import normalize, clamp
from utils.filters import gaussian_blur


def generate_sediment_map(heightmap: np.ndarray,
                          hydraulic_sediment: np.ndarray) -> np.ndarray:
    """
    Generate final sediment accumulation map.
    """
    sediment = hydraulic_sediment.copy()
    sediment = gaussian_blur(sediment, sigma=1.5)
    return normalize(sediment)
