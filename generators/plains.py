"""
generators/plains.py — Générateur de plaines et bassins.

Crée :
  - Plaines alluviales
  - Bassins sédimentaires
  - Smooth lowlands
"""

import numpy as np

from config import WorldConfig
from utils.noise import SimplexNoise, make_grid, fbm
from utils.math_utils import normalize, smoothstep, clamp
from utils.filters import gaussian_blur


def generate_plains(config: WorldConfig,
                    heightmap: np.ndarray) -> np.ndarray:
    """
    Flatten and smooth lowlands into natural alluvial plains and basins.
    """
    res = config.resolution
    seed = config.sub_seed("plains")

    print("  [plains] Generating lowlands and plains...")
    result = heightmap.copy()

    # Target low/mid elevations (avoid mountains and oceans)
    plains_mask = (heightmap > config.sea_level + 0.02) & (heightmap < 0.55)
    plains_weight = smoothstep(config.sea_level, config.sea_level + 0.05, heightmap) * \
                    (1.0 - smoothstep(0.45, 0.60, heightmap))

    # Low frequency smooth micro-relief
    noise = SimplexNoise(seed)
    xs, ys = make_grid(res, frequency=3.0)
    micro_relief = fbm(noise, xs, ys, octaves=3, frequency=3.0, persistence=0.3) * 0.03

    # Flatten terrain towards average level of plains
    blurred = gaussian_blur(heightmap, sigma=res / 40.0)

    # Flatten lowlands
    result = result * (1.0 - plains_weight * 0.6) + blurred * (plains_weight * 0.6) + micro_relief * plains_weight

    result = clamp(result, 0.0, 1.0)
    print("  [plains] Done.")
    return result
