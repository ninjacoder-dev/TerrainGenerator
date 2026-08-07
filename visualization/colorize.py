"""
visualization/colorize.py — Coloration de terrain par biomes / hypsométrie.
"""

import numpy as np
from biome.biome_generator import BIOME_COLORS, BiomeType


def colorize_by_biome(biome_map: np.ndarray) -> np.ndarray:
    """Map biome indices to RGB color image array in [0, 255]."""
    h, w = biome_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for b_type, color in BIOME_COLORS.items():
        mask = biome_map == b_type
        rgb[mask] = color

    return rgb
