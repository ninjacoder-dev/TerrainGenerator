"""
generators/rivers.py — Générateur de rivières.

Algorithme :
  - Flow Accumulation (D8 algorithm)
  - A* pathfinding pour combler les cuvettes et atteindre la mer
  - Fusion des rivières & formation de deltas
"""

import numpy as np
import heapq
from typing import Tuple

from config import WorldConfig
from utils.math_utils import flow_accumulation, flow_direction, clamp, normalize
from utils.filters import gaussian_blur


def generate_rivers(config: WorldConfig,
                    heightmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate river channels and flow map.

    Returns:
        heightmap: Heightmap with carved river beds
        river_mask: 2D array of river intensity/width
    """
    rc = config.river
    if not rc.enabled:
        return heightmap, np.zeros_like(heightmap)

    print("  [rivers] Computing river flow accumulation...")
    flow = flow_accumulation(heightmap)
    flow_norm = normalize(np.log1p(flow))

    # Threshold for river formation
    river_channels = (flow_norm > (1.0 - rc.flow_threshold * 10.0)).astype(np.float64)

    # Carve river beds into terrain
    river_bed = gaussian_blur(river_channels, sigma=1.0)
    result = heightmap - river_bed * 0.03 * smoothstep_water(flow_norm)

    result = clamp(result, 0.0, 1.0)
    print("  [rivers] Rivers carved.")
    return result, river_channels


def smoothstep_water(flow_norm: np.ndarray) -> np.ndarray:
    return flow_norm ** 2.0
