"""
erosion/thermal.py — Érosion thermique.

Simule l'éboulement des falaises et pentes raides dépassant l'angle de repos (talus angle).
Les matériaux glissent du haut vers le bas jusqu'à ce que la pente se stabilise.
"""

import numpy as np
from config import WorldConfig
from utils.math_utils import clamp
from utils.threading import njit, HAS_NUMBA


@njit
def _thermal_erosion_step_numba(heightmap: np.ndarray,
                                talus_angle: float,
                                erosion_rate: float) -> np.ndarray:
    """Numba-accelerated thermal erosion single iteration."""
    h, w = heightmap.shape
    new_h = heightmap.copy()

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            current = heightmap[y, x]

            # Compare with 4 neighbors
            max_diff = 0.0
            target_ny, target_nx = y, x

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                diff = current - heightmap[ny, nx]
                if diff > max_diff:
                    max_diff = diff
                    target_ny, target_nx = ny, nx

            if max_diff > talus_angle:
                delta = (max_diff - talus_angle) * 0.5 * erosion_rate
                new_h[y, x] -= delta
                new_h[target_ny, target_nx] += delta

    return new_h


def _thermal_erosion_step_numpy(heightmap: np.ndarray,
                                talus_angle: float,
                                erosion_rate: float) -> np.ndarray:
    """Pure numpy fallback for thermal erosion."""
    h, w = heightmap.shape
    new_h = heightmap.copy()

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rolled = np.roll(np.roll(heightmap, dy, axis=0), dx, axis=1)
        diff = heightmap - rolled

        # Mask where slope exceeds talus threshold
        mask = (diff > talus_angle) & (diff > 0)
        delta = (diff - talus_angle) * 0.25 * erosion_rate * mask

        new_h -= delta
        new_h += np.roll(np.roll(delta, -dy, axis=0), -dx, axis=1)

    return new_h


def simulate_thermal_erosion(config: WorldConfig,
                            heightmap: np.ndarray) -> np.ndarray:
    """
    Apply thermal erosion across multiple iterations.
    """
    tc = config.thermal_erosion
    if not tc.enabled:
        return heightmap

    print(f"  [erosion] Running thermal erosion ({tc.iterations} iterations)...")
    result = heightmap.copy()

    step_func = _thermal_erosion_step_numba if HAS_NUMBA else _thermal_erosion_step_numpy

    for i in range(tc.iterations):
        result = step_func(result, tc.talus_angle / config.resolution, tc.erosion_rate)

    result = clamp(result, 0.0, 1.0)
    print("  [erosion] Thermal erosion finished.")
    return result
