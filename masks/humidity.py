"""
masks/humidity.py — Simulation de l'humidité.

Facteurs :
  - Évaporation près des étendues d'eau
  - Vents dominants apportant la pluie
  - Ombres pluviométriques (rain shadows) derrière les montagnes
"""

import numpy as np
from scipy import ndimage
from typing import Optional

from config import WorldConfig
from utils.math_utils import normalize, clamp, slope_map
from utils.noise import SimplexNoise, make_grid, fbm


def generate_humidity_mask(config: WorldConfig,
                           heightmap: np.ndarray,
                           water_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate moisture map taking wind, rain shadows, and water proximity into account.
    """
    res = config.resolution
    hc = config.humidity
    seed = config.sub_seed("humidity")

    print("  [masks] Computing humidity map...")

    # Base humidity from noise
    noise = SimplexNoise(seed)
    xs, ys = make_grid(res, frequency=1.5)
    base_humidity = fbm(noise, xs, ys, octaves=4, frequency=1.5, persistence=0.5)
    base_humidity = normalize(base_humidity)

    # Moisture boosted near water bodies
    if water_mask is not None:
        water_proximity = ndimage.gaussian_filter(water_mask, sigma=res / 20.0)
        base_humidity += water_proximity * hc.evaporation_rate

    # Rain shadow effect: mountains block moisture carried by wind
    rad = np.radians(hc.wind_direction)
    dx_wind = np.cos(rad)
    dy_wind = np.sin(rad)

    mountain_barrier = clamp(heightmap - config.sea_level, 0.0, 1.0)
    shadow = np.zeros_like(heightmap)

    # Propagate shadow downwind
    for i in range(1, 20):
        shifted = np.roll(np.roll(mountain_barrier, int(i * dy_wind), axis=0), int(i * dx_wind), axis=1)
        shadow = np.maximum(shadow, shifted * (1.0 - i * 0.04))

    humidity = base_humidity - shadow * hc.rain_shadow_strength
    humidity = clamp(humidity, 0.0, 1.0)

    print("  [masks] Humidity map finished.")
    return humidity
