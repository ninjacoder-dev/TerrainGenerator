"""
masks/biome.py — Wrapper pour la carte de biomes.
"""

import numpy as np
from biome.biome_generator import classify_biomes


def generate_biome_mask(heightmap: np.ndarray,
                        humidity: np.ndarray,
                        temperature: np.ndarray,
                        sea_level: float) -> np.ndarray:
    """Generate 2D biome classification grid."""
    return classify_biomes(heightmap, humidity, temperature, sea_level)
