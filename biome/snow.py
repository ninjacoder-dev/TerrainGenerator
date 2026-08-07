"""
biome/snow.py — Placement du manteau neigeux.
"""

import numpy as np
from biome.biome_generator import BiomeType
from utils.math_utils import clamp


def generate_snow_mask(heightmap: np.ndarray,
                       temperature: np.ndarray,
                       curvature: np.ndarray) -> np.ndarray:
    """Generate snow coverage mask (1 = snow covered, 0 = bare)."""
    # Cold temperatures or high altitudes
    cold_snow = clamp((0.25 - temperature) * 4.0, 0.0, 1.0)
    altitude_snow = clamp((heightmap - 0.72) * 4.0, 0.0, 1.0)

    snow = np.maximum(cold_snow, altitude_snow)

    # Snow accumulates more in concave hollows
    snow += (curvature - 0.5) * 0.2 * (snow > 0.1)

    return clamp(snow, 0.0, 1.0)
