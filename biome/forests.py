"""
biome/forests.py — Placement des forêts.

Density & distribution based on biomes, slope, and moisture.
"""

import numpy as np
from biome.biome_generator import BiomeType
from utils.math_utils import clamp


def generate_forest_mask(biome_map: np.ndarray,
                         slope: np.ndarray,
                         humidity: np.ndarray) -> np.ndarray:
    """Generate forest density mask (0 = no trees, 1 = dense forest)."""
    h, w = biome_map.shape
    forest = np.zeros((h, w), dtype=np.float64)

    # Biomes supporting forests
    forest_biomes = (biome_map == BiomeType.FOREST) | \
                    (biome_map == BiomeType.TAIGA) | \
                    (biome_map == BiomeType.JUNGLE)

    forest[forest_biomes] = 0.8

    # Modulated by humidity
    forest *= humidity

    # Trees cannot grow on extreme slopes (>0.45)
    gentle_slope = clamp((0.45 - slope) * 3.0, 0.0, 1.0)
    forest *= gentle_slope

    return clamp(forest, 0.0, 1.0)
