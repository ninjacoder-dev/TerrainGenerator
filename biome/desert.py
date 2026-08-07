"""
biome/desert.py — Masque du sol désertique.
"""

import numpy as np
from biome.biome_generator import BiomeType


def generate_desert_mask(biome_map: np.ndarray) -> np.ndarray:
    """Generate arid sand/desert ground mask."""
    return (biome_map == BiomeType.DESERT).astype(np.float64)
