"""
generators/base_noise.py — Génération du relief de base par empilage de bruit.

Combine plusieurs couches de bruit (OpenSimplex, Ridged, fBm, Billow)
à différentes fréquences pour créer le relief fondamental du terrain.
"""

import numpy as np
from typing import Optional

from config import WorldConfig, NoiseConfig, WarpConfig
from utils.noise import (
    SimplexNoise, make_grid, fbm, ridged, billow,
    domain_warp, noise_stack
)
from utils.math_utils import normalize, smoothstep, blend, power_curve


def generate_base_noise(config: WorldConfig,
                        continent_mask: Optional[np.ndarray] = None
                        ) -> np.ndarray:
    """
    Generate the base terrain relief from layered noise.

    This is the foundational heightmap before feature generators
    (mountains, volcanos, etc.) and erosion are applied.

    Pipeline:
        1. Create noise coordinate grid
        2. Apply domain warping (if enabled)
        3. Stack multiple noise layers (fBm + Ridged + Billow)
        4. Modulate by continent mask (if provided)
        5. Normalize

    Parameters:
        config: World configuration.
        continent_mask: Optional continental mask (0 = ocean, 1 = land).

    Returns:
        2D heightmap array, normalized to [0, 1].
    """
    res = config.resolution
    seed = config.sub_seed("base_noise")
    nc = config.noise
    wc = config.warp

    print("  [base_noise] Generating coordinate grid...")
    xs, ys = make_grid(res, frequency=nc.frequency)

    # ----- Domain Warping -----
    if wc.enabled:
        print(f"  [base_noise] Applying domain warping (strength={wc.strength}, "
              f"iterations={wc.iterations})...")
        warp_noise = SimplexNoise(seed + 999)
        xs, ys = domain_warp(
            warp_noise, xs, ys,
            strength=wc.strength,
            warp_octaves=wc.octaves,
            warp_frequency=wc.frequency,
            iterations=wc.iterations
        )

    # ----- Noise Stack -----
    print(f"  [base_noise] Computing noise stack ({nc.octaves} octaves)...")

    layers = [
        {
            "type": "fbm",
            "weight": 0.40,
            "octaves": nc.octaves,
            "frequency": nc.frequency,
            "persistence": nc.persistence,
            "lacunarity": nc.lacunarity,
            "amplitude": nc.amplitude,
        },
        {
            "type": "ridged",
            "weight": 0.35,
            "octaves": max(4, nc.octaves - 2),
            "frequency": nc.frequency * 1.5,
            "persistence": nc.persistence * 0.9,
            "lacunarity": nc.lacunarity,
            "amplitude": nc.amplitude,
        },
        {
            "type": "billow",
            "weight": 0.15,
            "octaves": max(3, nc.octaves - 3),
            "frequency": nc.frequency * 2.0,
            "persistence": nc.persistence * 0.8,
            "lacunarity": nc.lacunarity,
            "amplitude": nc.amplitude,
        },
        {
            "type": "fbm",
            "weight": 0.10,
            "octaves": max(2, nc.octaves - 4),
            "frequency": nc.frequency * 4.0,
            "persistence": nc.persistence * 0.7,
            "lacunarity": nc.lacunarity,
            "amplitude": nc.amplitude * 0.5,
        },
    ]

    heightmap = noise_stack(seed, xs, ys, layers)

    # ----- Apply continent mask -----
    if continent_mask is not None:
        print("  [base_noise] Modulating with continent mask...")
        # Continental areas get higher terrain, oceans get depressed
        ocean_floor = _generate_ocean_floor(seed + 500, res, nc)
        land_terrain = heightmap * 0.6 + 0.4  # Raise land

        # Smooth transition at coast
        coast_blend = smoothstep(0.3, 0.7, continent_mask)
        heightmap = blend(ocean_floor, land_terrain, coast_blend)

    # ----- Normalize -----
    heightmap = normalize(heightmap, 0.0, 1.0)

    print("  [base_noise] Done.")
    return heightmap


def _generate_ocean_floor(seed: int, resolution: int,
                          nc: NoiseConfig) -> np.ndarray:
    """Generate subtle ocean floor topography."""
    xs, ys = make_grid(resolution, frequency=nc.frequency * 0.5)
    noise = SimplexNoise(seed)

    floor = fbm(noise, xs, ys,
                octaves=4,
                frequency=nc.frequency * 0.5,
                persistence=0.4,
                amplitude=0.15)

    # Ocean floor should be low
    floor = floor * 0.15 + 0.15
    return floor
