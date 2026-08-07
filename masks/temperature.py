"""
masks/temperature.py — Carte de température.

Calculée avec :
  - Latitude (gradient Nord-Sud)
  - Altitude (refroidissement avec l'altitude / lapse rate)
  - Proximité des océans (modération thermique)
"""

import numpy as np

from config import WorldConfig
from utils.math_utils import normalize, clamp


def generate_temperature_mask(config: WorldConfig,
                              heightmap: np.ndarray) -> np.ndarray:
    """
    Generate temperature map in normalized scale (0 = freezing, 1 = hot).
    """
    res = config.resolution
    tc = config.temperature

    print("  [masks] Computing temperature map...")

    # Latitude gradient (warmer at equator/center, cooler at poles/edges)
    lin = np.linspace(0, 1, res)
    y_grid, _ = np.meshgrid(lin, lin)
    latitude_temp = 1.0 - np.abs(y_grid - 0.5) * 2.0 * tc.latitude_gradient

    # Altitude lapse rate (higher = colder)
    altitude_cooling = heightmap * (tc.altitude_lapse_rate / 30.0)

    temperature = latitude_temp - altitude_cooling
    temperature = clamp(temperature, 0.0, 1.0)

    print("  [masks] Temperature map finished.")
    return temperature
