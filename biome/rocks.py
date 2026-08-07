"""
biome/rocks.py — Placement des rochers et falaises.

Masque basé sur :
  - Pente forte (falaises)
  - Altitude élevée (scree / chaos rocheux)
  - Absence de biomes très humides
"""

import numpy as np
from utils.math_utils import clamp, slope_map
from utils.noise import SimplexNoise, make_grid, fbm


def generate_rock_mask(heightmap: np.ndarray,
                       slope: np.ndarray,
                       sea_level: float,
                       seed: int = 0) -> np.ndarray:
    """Generate rock / cliff distribution mask."""
    res = heightmap.shape[0]
    land = heightmap > sea_level

    # Steep slopes are bare rock
    steep_rock = clamp((slope - 0.25) * 3.0, 0.0, 1.0)

    # High altitude peaks are rocky
    high_rock = clamp((heightmap - 0.7) * 3.0, 0.0, 1.0)

    # Add noise for scattered boulders
    noise = SimplexNoise(seed)
    xs, ys = make_grid(res, frequency=12.0)
    rock_noise = fbm(noise, xs, ys, octaves=3, frequency=12.0, persistence=0.5)
    scattered = clamp((rock_noise - 0.3) * 2.0, 0.0, 1.0) * (heightmap > sea_level + 0.1)

    rock_mask = clamp(steep_rock + high_rock * 0.7 + scattered * 0.3, 0.0, 1.0) * land
    return rock_mask
