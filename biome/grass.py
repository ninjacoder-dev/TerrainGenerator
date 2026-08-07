"""
biome/grass.py — Placement du tapis d'herbe.
"""

import numpy as np
from biome.biome_generator import BiomeType
from utils.math_utils import clamp


def generate_grass_mask(biome_map: np.ndarray,
                        slope: np.ndarray) -> np.ndarray:
    """Generate grass coverage mask."""
    grass_biomes = (biome_map == BiomeType.GRASSLAND) | \
                   (biome_map == BiomeType.SAVANNA) | \
                   (biome_map == BiomeType.FOREST)

    grass = grass_biomes.astype(np.float64) * 0.9

    # Moderate slopes only
    gentle_slope = clamp((0.35 - slope) * 3.0, 0.0, 1.0)
    return clamp(grass * gentle_slope, 0.0, 1.0)
