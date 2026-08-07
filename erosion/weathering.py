"""
erosion/weathering.py — Altération mécanique et chimique.

Subdivise et adoucit la roche exposée à l'air et à la pluie.
"""

import numpy as np

from config import WorldConfig
from utils.math_utils import clamp
from utils.filters import gaussian_blur, sharpen


def simulate_weathering(config: WorldConfig,
                        heightmap: np.ndarray) -> np.ndarray:
    """
    Simulate subtle rock weathering / aging.
    """
    print("  [erosion] Applying surface weathering...")
    # Micro noise weathering
    weathered = gaussian_blur(heightmap, sigma=1.0) * 0.2 + heightmap * 0.8
    return clamp(weathered, 0.0, 1.0)
