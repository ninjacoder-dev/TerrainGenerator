"""
generators/lakes.py — Générateur de lacs.

Créés automatiquement dans les cuvettes/dépressions fermées croisant les rivières.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional

from config import WorldConfig
from utils.math_utils import clamp


def generate_lakes(config: WorldConfig,
                   heightmap: np.ndarray,
                   river_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect depressions (pits/sinks) and fill them with lake water.

    Returns:
        heightmap: Heightmap with leveled lake surfaces
        lake_mask: 2D array of lake water bodies
    """
    lc = config.lake
    if not lc.enabled:
        return heightmap, np.zeros_like(heightmap)

    print("  [lakes] Detecting terrain basins and forming lakes...")
    h, w = heightmap.shape

    # Morphological reconstruction to find sinks (pits lower than surroundings)
    seed = heightmap + 0.05
    seed[0, :] = heightmap[0, :]
    seed[-1, :] = heightmap[-1, :]
    seed[:, 0] = heightmap[:, 0]
    seed[:, -1] = heightmap[:, -1]

    # Filled terrain
    filled = ndimage.grey_erosion(seed, size=(3, 3))
    sinks = filled - heightmap

    # Filter small noisy sinks
    lake_mask = (sinks > lc.fill_depth * 0.1).astype(np.float64)
    lake_mask = ndimage.binary_opening(lake_mask, structure=np.ones((3, 3))).astype(np.float64)

    # Flatten lake water surface
    result = heightmap.copy()
    result[lake_mask > 0] = filled[lake_mask > 0]

    result = clamp(result, 0.0, 1.0)
    print("  [lakes] Lakes formed.")
    return result, lake_mask
