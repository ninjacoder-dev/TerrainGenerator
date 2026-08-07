"""
generators/glaciers.py — Générateur de glaciers.

Les glaciers :
  - Se forment à haute altitude
  - S'écoulent le long des pentes
  - Rabotent la roche et déposent des moraines
"""

import numpy as np
from scipy import ndimage
from typing import Tuple

from config import WorldConfig
from utils.math_utils import normalize, smoothstep, clamp, slope_map
from utils.filters import gaussian_blur


def generate_glaciers(config: WorldConfig,
                      heightmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate ice coverage and moraine deposits.

    Returns:
        heightmap: Modified heightmap with moraines
        glacier_mask: 2D array of ice thickness/coverage
    """
    res = config.resolution
    gc = config.glacier

    print("  [glaciers] Simulating glaciers and moraine deposits...")

    # High altitude accumulation zone
    accumulation = smoothstep(gc.altitude_threshold - 0.05, gc.altitude_threshold + 0.1, heightmap)

    # Ice flows downhill (downward blur + accumulation propagation)
    glacier_ice = accumulation.copy()
    slopes = slope_map(heightmap)

    # Downslope flow simulation passes
    for _ in range(15):
        flow = ndimage.gaussian_filter(glacier_ice, sigma=res / 150.0)
        # Ice flows more easily on steep slopes downwards
        glacier_ice = np.maximum(glacier_ice, flow * (1.0 - slopes * 0.5) * 0.95)

    glacier_ice = clamp(glacier_ice, 0.0, 1.0)

    # Moraine ridges formed at the glacier terminus
    terminus_edges = ndimage.sobel(glacier_ice)
    moraines = np.abs(terminus_edges) * (glacier_ice < 0.2)
    moraines = gaussian_blur(moraines, sigma=2.0)

    result = heightmap + moraines * gc.moraine_height

    result = clamp(result, 0.0, 1.0)
    print("  [glaciers] Done.")
    return result, glacier_ice
