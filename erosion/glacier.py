"""
erosion/glacier.py — Érosion glaciaire.

Les glaciers :
  - Rabotent les vallées (abrasion)
  - Arrachent la roche (plucking)
  - Façonnent des vallées en U
"""

import numpy as np

from config import WorldConfig
from utils.math_utils import clamp
from utils.filters import gaussian_blur


def simulate_glacial_erosion(config: WorldConfig,
                             heightmap: np.ndarray,
                             glacier_mask: np.ndarray) -> np.ndarray:
    """
    Apply glacial carving (abrasion + plucking) where glaciers exist.
    """
    ge = config.glacial_erosion
    if not ge.enabled or glacier_mask is None or np.max(glacier_mask) == 0:
        return heightmap

    print("  [erosion] Applying glacial erosion...")
    result = heightmap.copy()

    # Abrasion: glacier weight smooths and carves valley bed
    carving = gaussian_blur(glacier_mask, sigma=3.0) * ge.abrasion_rate * 0.05
    result -= carving

    result = clamp(result, 0.0, 1.0)
    print("  [erosion] Glacial erosion finished.")
    return result
