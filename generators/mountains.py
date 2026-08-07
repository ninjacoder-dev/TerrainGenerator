"""
generators/mountains.py — Générateur de montagnes.

Types de montagnes :
  - Alpes (ridged multifractal, pentes raides)
  - Himalaya (très hautes, massives)
  - Rocheuses (rugueuses, fractales)
  - Volcaniques (cônes isolés)
  - Plateaux (sommets plats)
  - Collines (douces, arrondies)
"""

import numpy as np
from typing import Optional

from config import WorldConfig
from utils.noise import (
    SimplexNoise, make_grid, fbm, ridged, billow, domain_warp
)
from utils.math_utils import normalize, smoothstep, blend, clamp, power_curve
from utils.filters import gaussian_blur


def generate_mountains(config: WorldConfig,
                       heightmap: np.ndarray,
                       boundary_map: Optional[np.ndarray] = None
                       ) -> np.ndarray:
    """
    Add mountain features to an existing heightmap.

    Mountains are placed preferentially at tectonic boundaries
    (if boundary_map is provided) and in high-altitude areas.
    """
    res = config.resolution
    mc = config.mountains
    seed = config.sub_seed("mountains")

    print("  [mountains] Generating mountain features...")

    result = heightmap.copy()

    # ----- Mountain placement mask -----
    # Mountains prefer high terrain and tectonic boundaries
    placement = smoothstep(0.4, 0.7, heightmap)
    if boundary_map is not None:
        placement = clamp(placement + boundary_map * 0.5, 0.0, 1.0)

    # ----- Alpine mountains (sharp ridges) -----
    print("  [mountains] Generating alpine ridges...")
    noise = SimplexNoise(seed)
    xs, ys = make_grid(res, frequency=2.0)

    # Domain warp for organic shapes
    warp_noise = SimplexNoise(seed + 50)
    wxs, wys = domain_warp(warp_noise, xs, ys, strength=60.0,
                           warp_octaves=3, warp_frequency=1.0, iterations=2)

    alpine = ridged(noise, wxs, wys,
                    octaves=8, frequency=2.0,
                    persistence=0.55, lacunarity=2.1,
                    offset=1.0, gain=2.0)
    alpine = normalize(alpine)
    alpine = power_curve(alpine, mc.ridge_sharpness)

    # ----- Rocky mountains (rough, fractured) -----
    print("  [mountains] Generating rocky terrain...")
    rocky_noise = SimplexNoise(seed + 100)
    rocky_xs, rocky_ys = make_grid(res, frequency=3.0)

    rocky = fbm(rocky_noise, rocky_xs, rocky_ys,
                octaves=10, frequency=3.0,
                persistence=0.6, lacunarity=2.0)
    rocky_detail = ridged(SimplexNoise(seed + 150), rocky_xs, rocky_ys,
                          octaves=6, frequency=6.0,
                          persistence=0.5, lacunarity=2.2)
    rocky = normalize(rocky * 0.6 + rocky_detail * 0.4)

    # ----- Plateaus (flat-topped) -----
    print("  [mountains] Generating plateaus...")
    plateau_noise = SimplexNoise(seed + 200)
    plateau_xs, plateau_ys = make_grid(res, frequency=1.0)
    plateau_base = fbm(plateau_noise, plateau_xs, plateau_ys,
                       octaves=4, frequency=1.0, persistence=0.4)
    plateau_base = normalize(plateau_base)

    # Flatten tops using smoothstep
    plateau = smoothstep(0.4, 0.6, plateau_base)
    plateau_mask = smoothstep(0.6, 0.8, plateau_base) * 0.3  # Where plateaus form

    # ----- Hills (soft, rounded) -----
    print("  [mountains] Generating hills...")
    hill_noise = SimplexNoise(seed + 300)
    hill_xs, hill_ys = make_grid(res, frequency=2.5)
    hills = billow(hill_noise, hill_xs, hill_ys,
                   octaves=5, frequency=2.5,
                   persistence=0.4, lacunarity=2.0)
    hills = normalize(hills) * 0.3  # Low amplitude

    # ----- Combine mountain types -----
    print("  [mountains] Combining mountain types...")

    # Create type distribution masks
    type_noise = SimplexNoise(seed + 400)
    type_xs, type_ys = make_grid(res, frequency=0.8)
    type_map = fbm(type_noise, type_xs, type_ys,
                   octaves=3, frequency=0.8, persistence=0.5)
    type_map = normalize(type_map)

    # Blend based on type map regions
    mountains = np.zeros_like(heightmap)
    mountains += alpine * smoothstep(0.5, 0.8, type_map) * 0.5
    mountains += rocky * smoothstep(0.2, 0.5, type_map) * (1.0 - smoothstep(0.5, 0.8, type_map)) * 0.4
    mountains += hills * (1.0 - smoothstep(0.3, 0.6, type_map)) * 0.3
    mountains += plateau * plateau_mask

    # Apply placement mask and height scaling
    mountains = mountains * placement * mc.max_height

    # Steep slopes
    mountains = power_curve(normalize(mountains), 1.0 / mc.slope_steepness)

    # Add to heightmap
    result = result + mountains * 0.4
    result = normalize(result)

    print("  [mountains] Done.")
    return result
