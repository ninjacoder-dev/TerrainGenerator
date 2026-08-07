"""
generators/ocean.py — Gestion des océans, côtes et marécages.

Gère :
  - Niveau de la mer (sea level)
  - Marécages littoraux (coastal wetlands)
  - Gradient des côtes
"""

import numpy as np
from typing import Tuple
from config import WorldConfig
from utils.math_utils import clamp, smoothstep


def generate_ocean(config: WorldConfig,
                   heightmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure sea level flattening and coastal shelf transitions.

    Returns:
        heightmap: Heightmap with sea level floor applied
        ocean_mask: Binary mask of ocean areas
    """
    sea_level = config.sea_level
    ocean_mask = (heightmap < sea_level).astype(np.float64)

    # Flatten deep seabed slightly for smooth ocean floors
    result = heightmap.copy()
    underwater = result < sea_level
    result[underwater] = sea_level * 0.95 + result[underwater] * 0.05

    return clamp(result, 0.0, 1.0), ocean_mask
