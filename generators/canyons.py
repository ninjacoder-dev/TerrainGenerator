"""
generators/canyons.py — Générateur de canyons.

Utilise :
  - Bruit sinueux
  - Masque de pente
  - Découpe en terrasses/marches d'escalier
  - Érosion encaissée (ex: Grand Canyon)
"""

import numpy as np
from typing import Optional

from config import WorldConfig
from utils.noise import SimplexNoise, make_grid, fbm, ridged, domain_warp
from utils.math_utils import normalize, smoothstep, clamp, slope_map, terrace
from utils.filters import gaussian_blur


def generate_canyons(config: WorldConfig,
                     heightmap: np.ndarray) -> np.ndarray:
    """
    Carve canyon networks into plateau/high terrain.
    """
    res = config.resolution
    cc = config.canyon
    seed = config.sub_seed("canyons")

    print("  [canyons] Carving canyons...")
    result = heightmap.copy()

    # Canyons require plateau or elevated terrain
    canyon_terrain_mask = smoothstep(0.4, 0.7, heightmap)

    noise = SimplexNoise(seed)
    xs, ys = make_grid(res, frequency=1.2)

    # Domain warp for sinuous river/canyon paths
    warp_noise = SimplexNoise(seed + 42)
    wxs, wys = domain_warp(
        warp_noise, xs, ys,
        strength=cc.sinuosity * 80.0,
        warp_octaves=4,
        warp_frequency=0.8,
        iterations=3
    )

    # Create network of narrow fractures using ridged noise
    canyon_network = ridged(
        noise, wxs, wys,
        octaves=6, frequency=1.5,
        persistence=0.5, lacunarity=2.0
    )
    canyon_network = normalize(canyon_network)

    # Invert so ridges become deep slots
    canyon_mask = 1.0 - canyon_network
    canyon_mask = smoothstep(0.7, 0.95, canyon_mask)

    # Terracing effect (layered rock strata like Grand Canyon)
    terraced_depth = terrace(canyon_mask, levels=6)
    canyon_mask = blend_terrace(canyon_mask, terraced_depth, cc.wall_steepness)

    # Carve into heightmap
    carve_amount = canyon_mask * cc.depth * canyon_terrain_mask
    result -= carve_amount

    result = clamp(result, 0.0, 1.0)
    print("  [canyons] Done.")
    return result


def blend_terrace(mask: np.ndarray, terraced: np.ndarray, steepness: float) -> np.ndarray:
    """Combine smooth depth with terraced step depth."""
    return mask * (1.0 - steepness * 0.5) + terraced * (steepness * 0.5)
