"""
masks/water.py — Masque d'eau.

Distingue :
  - Océans (sous le sea level)
  - Rivières
  - Lacs
"""

import numpy as np
from typing import Optional
from utils.math_utils import clamp


def generate_water_mask(heightmap: np.ndarray,
                        sea_level: float,
                        river_mask: Optional[np.ndarray] = None,
                        lake_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Generate composite binary/continuous water mask (1 = water, 0 = land)."""
    ocean = (heightmap <= sea_level).astype(np.float64)
    water = ocean.copy()

    if river_mask is not None:
        water = np.maximum(water, river_mask)
    if lake_mask is not None:
        water = np.maximum(water, lake_mask)

    return clamp(water, 0.0, 1.0)
