"""
masks/altitude.py — Carte d'altitude.

Normalise la hauteur de 0 (mer) à 1 (sommet).
"""

import numpy as np
from utils.math_utils import normalize


def generate_altitude_mask(heightmap: np.ndarray) -> np.ndarray:
    """Return normalized altitude map."""
    return normalize(heightmap)
