"""
visualization/hillshade.py — Hillshade et ombrage topographique.
"""

import numpy as np
from utils.math_utils import clamp


def generate_hillshade(heightmap: np.ndarray,
                       azimuth: float = 315.0,
                       altitude: float = 45.0) -> np.ndarray:
    """
    Generate classical relief hillshade array in [0.0, 1.0].
    """
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    dy, dx = np.gradient(heightmap)

    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(-dx, dy)

    shaded = np.sin(alt_rad) * np.cos(slope) + \
             np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)

    return clamp(shaded, 0.0, 1.0)
