"""
erosion/wind.py — Érosion éolienne.

Simule :
  - Abrasion du vent sur les sommets et arêtes exposées
  - Transport de particules de sable
  - Dépôt dans les zones abritées du vent
"""

import numpy as np
from scipy import ndimage

from config import WorldConfig
from utils.math_utils import clamp, slope_map


def simulate_wind_erosion(config: WorldConfig,
                         heightmap: np.ndarray) -> np.ndarray:
    """
    Apply wind abrasion and particle deposition based on wind vector field.
    """
    wc = config.wind_erosion
    if not wc.enabled:
        return heightmap

    print(f"  [erosion] Simulating wind erosion ({wc.iterations} iterations)...")
    result = heightmap.copy()

    rad = np.radians(wc.wind_direction)
    dx_wind = np.cos(rad)
    dy_wind = np.sin(rad)

    # Exposed ridges (high curvature / high elevation windward)
    slopes = slope_map(heightmap)
    dy, dx = np.gradient(heightmap)

    # Windward exposure: dot product of slope gradient and wind direction
    exposure = clamp(dx * dx_wind + dy * dy_wind, 0.0, 1.0) * slopes

    # Abrasion removes material from windward slopes
    abrasion = exposure * wc.abrasion_rate * 0.02
    result -= abrasion

    # Deposition in leeward shadow zones
    leeward = clamp(-(dx * dx_wind + dy * dy_wind), 0.0, 1.0)
    deposition = ndimage.gaussian_filter(leeward, sigma=2.0) * wc.transport_rate * 0.015
    result += deposition

    result = clamp(result, 0.0, 1.0)
    print("  [erosion] Wind erosion finished.")
    return result
