"""
erosion/hydraulic.py — Érosion hydraulique par simulation de gouttelettes.

Chaque goutte :
  1. Naît à une position aléatoire
  2. Descend le long de la pente
  3. Accélère et gagne en vitesse
  4. Érode le terrain selon sa capacité
  5. Transporte et dépose des sédiments
  6. S'évapore progressivement
"""

import numpy as np
from typing import Tuple

from config import WorldConfig
from utils.math_utils import clamp
from utils.threading import njit, HAS_NUMBA


@njit
def _simulate_droplets_numba(heightmap: np.ndarray,
                             num_droplets: int,
                             seed: int,
                             inertia: float,
                             capacity_factor: float,
                             min_slope: float,
                             erosion_rate: float,
                             deposition_rate: float,
                             evaporation_rate: float,
                             gravity: float,
                             max_lifetime: int,
                             erosion_radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba accelerated droplet simulation engine.
    Returns (modified_heightmap, sediment_map).
    """
    h, w = heightmap.shape
    heights = heightmap.copy()
    sediment_map = np.zeros((h, w), dtype=np.float64)

    np.random.seed(seed)

    for _ in range(num_droplets):
        # Random initial position
        px = np.random.uniform(1.0, w - 2.0)
        py = np.random.uniform(1.0, h - 2.0)

        dir_x = 0.0
        dir_y = 0.0
        speed = 1.0
        water = 1.0
        sediment = 0.0

        for _ in range(max_lifetime):
            ix = int(px)
            iy = int(py)

            # Calculate gradient via bilinear interpolation
            u = px - ix
            v = py - iy

            h00 = heights[iy, ix]
            h10 = heights[iy, ix + 1]
            h01 = heights[iy + 1, ix]
            h11 = heights[iy + 1, ix + 1]

            gx = (h10 - h00) * (1.0 - v) + (h11 - h01) * v
            gy = (h01 - h00) * (1.0 - u) + (h11 - h10) * u

            # Calculate height at current position
            current_height = h00 * (1 - u) * (1 - v) + h10 * u * (1 - v) + h01 * (1 - u) * v + h11 * u * v

            # Update direction with inertia
            dir_x = dir_x * inertia - gx * (1.0 - inertia)
            dir_y = dir_y * inertia - gy * (1.0 - inertia)

            # Normalize direction
            length = np.sqrt(dir_x * dir_x + dir_y * dir_y)
            if length != 0:
                dir_x /= length
                dir_y /= length
            else:
                dir_x = np.random.uniform(-1.0, 1.0)
                dir_y = np.random.uniform(-1.0, 1.0)

            # Move droplet
            new_px = px + dir_x
            new_py = py + dir_y

            # Check bounds
            if new_px < 1.0 or new_px >= w - 2.0 or new_py < 1.0 or new_py >= h - 2.0:
                break

            # Calculate new height
            n_ix = int(new_px)
            n_iy = int(new_py)
            nu = new_px - n_ix
            nv = new_py - n_iy
            new_height = heights[n_iy, n_ix] * (1 - nu) * (1 - nv) + \
                         heights[n_iy, n_ix + 1] * nu * (1 - nv) + \
                         heights[n_iy + 1, n_ix] * (1 - nu) * nv + \
                         heights[n_iy + 1, n_ix + 1] * nu * nv

            delta_h = new_height - current_height

            # Compute sediment capacity
            capacity = max(-delta_h, min_slope) * speed * water * capacity_factor

            if sediment > capacity or delta_h > 0:
                # Deposit sediment
                amount_to_deposit = (sediment - capacity) * deposition_rate if delta_h <= 0 else min(sediment, delta_h)
                sediment -= amount_to_deposit
                heights[iy, ix] += amount_to_deposit * (1 - u) * (1 - v)
                heights[iy, ix + 1] += amount_to_deposit * u * (1 - v)
                heights[iy + 1, ix] += amount_to_deposit * (1 - u) * v
                heights[iy + 1, ix + 1] += amount_to_deposit * u * v
                sediment_map[iy, ix] += amount_to_deposit
            else:
                # Erode terrain
                amount_to_erode = min((capacity - sediment) * erosion_rate, -delta_h)
                sediment += amount_to_erode
                heights[iy, ix] -= amount_to_erode * (1 - u) * (1 - v)
                heights[iy, ix + 1] -= amount_to_erode * u * (1 - v)
                heights[iy + 1, ix] -= amount_to_erode * (1 - u) * v
                heights[iy + 1, ix + 1] -= amount_to_erode * u * v

            # Update speed & water
            speed = np.sqrt(max(0.0, speed * speed + delta_h * gravity))
            water *= (1.0 - evaporation_rate)
            px = new_px
            py = new_py

    return heights, sediment_map


def simulate_hydraulic_erosion(config: WorldConfig,
                              heightmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run particle-based hydraulic erosion simulation.
    Returns (eroded_heightmap, sediment_map).
    """
    hc = config.hydraulic_erosion
    if not hc.enabled:
        return heightmap, np.zeros_like(heightmap)

    seed = config.sub_seed("hydraulic_erosion")
    print(f"  [erosion] Simulating hydraulic erosion ({hc.num_droplets:,} droplets)...")

    if HAS_NUMBA:
        eroded, sediment = _simulate_droplets_numba(
            heightmap, hc.num_droplets, seed,
            hc.inertia, hc.capacity_factor, hc.min_slope,
            hc.erosion_rate, hc.deposition_rate, hc.evaporation_rate,
            hc.gravity, hc.max_lifetime, hc.erosion_radius
        )
    else:
        # Fallback with reduced droplets for pure Python performance
        print("  [erosion] Note: Numba not detected; running fallback...")
        # When Numba is unavailable, njit is a no-op decorator, so the function is callable directly
        eroded, sediment = _simulate_droplets_numba(
            heightmap, min(hc.num_droplets, 20_000), seed,
            hc.inertia, hc.capacity_factor, hc.min_slope,
            hc.erosion_rate, hc.deposition_rate, hc.evaporation_rate,
            hc.gravity, hc.max_lifetime, hc.erosion_radius
        )

    eroded = clamp(eroded, 0.0, 1.0)
    print("  [erosion] Hydraulic erosion finished.")
    return eroded, sediment
