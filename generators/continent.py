"""
generators/continent.py — Génération du masque continental.

Décide de la répartition :
  - Océans
  - Continents
  - Archipels
  - Îles
  - Mers intérieures

Techniques : Worley Noise, OpenSimplex, Domain Warping, Distance Transform.
"""

import numpy as np
from typing import Optional, Tuple

from config import WorldConfig
from utils.noise import (
    SimplexNoise, make_grid, fbm, worley, domain_warp
)
from utils.math_utils import (
    normalize, smoothstep, blend, distance_transform,
    invert_distance_transform, clamp
)
from utils.filters import gaussian_blur


def generate_continent_mask(config: WorldConfig) -> np.ndarray:
    """
    Generate a continental mask determining land vs ocean.

    Output: 2D array where:
        0.0 = deep ocean
        0.0-0.3 = shallow ocean / shelf
        0.3-0.5 = coast / transition
        0.5-1.0 = land (low to high elevation areas)

    The algorithm:
        1. Create large-scale Worley noise for continent shapes
        2. Add OpenSimplex for natural variation
        3. Apply domain warping for organic coastlines
        4. Add small-scale noise for islands and archipelagos
        5. Smooth the coast with distance transform
        6. Threshold to create land/ocean separation
    """
    res = config.resolution
    cc = config.continent
    seed = config.sub_seed("continent")

    print(f"  [continent] Generating continent mask ({res}×{res})...")

    # ----- Step 1: Worley noise for continent shapes -----
    print("  [continent] Computing Worley noise for continent shapes...")
    worley_f1 = worley(
        res, num_points=8, seed=seed,
        distance_type="euclidean", return_type="F1"
    )
    worley_f2f1 = worley(
        res, num_points=8, seed=seed,
        distance_type="euclidean", return_type="F2-F1"
    )

    # Combine Worley patterns
    continent_base = (
        cc.worley_weight * (1.0 - worley_f1)
        + (1.0 - cc.worley_weight) * worley_f2f1
    )
    continent_base = normalize(continent_base)

    # ----- Step 2: OpenSimplex for natural variation -----
    print("  [continent] Adding OpenSimplex variation...")
    noise = SimplexNoise(seed + 100)
    xs, ys = make_grid(res, frequency=cc.continent_frequency)

    simplex_layer = fbm(
        noise, xs, ys,
        octaves=6, persistence=0.5, lacunarity=2.0,
        frequency=cc.continent_frequency
    )
    simplex_layer = normalize(simplex_layer)

    # Blend Worley and Simplex
    continent_base = blend(continent_base, simplex_layer, 0.5 * np.ones_like(continent_base))

    # ----- Step 3: Domain warping for organic coastlines -----
    print("  [continent] Applying domain warping to coastlines...")
    warp_noise = SimplexNoise(seed + 200)
    warped_xs, warped_ys = domain_warp(
        warp_noise, xs, ys,
        strength=40.0,
        warp_octaves=3,
        warp_frequency=0.8,
        iterations=2
    )

    warped_layer = fbm(
        noise, warped_xs, warped_ys,
        octaves=5, persistence=0.45, lacunarity=2.0,
        frequency=cc.continent_frequency
    )
    warped_layer = normalize(warped_layer)

    continent_base = blend(continent_base, warped_layer, 0.3 * np.ones_like(continent_base))

    # ----- Step 4: Islands and archipelagos -----
    print("  [continent] Adding islands and archipelagos...")
    island_noise = SimplexNoise(seed + 300)
    island_xs, island_ys = make_grid(res, frequency=cc.island_frequency)

    islands = fbm(
        island_noise, island_xs, island_ys,
        octaves=4, persistence=0.5, lacunarity=2.0,
        frequency=cc.island_frequency
    )
    islands = normalize(islands)

    # Islands appear only where the base is below sea level
    island_mask = smoothstep(0.3, 0.6, islands) * cc.island_density
    ocean_areas = 1.0 - smoothstep(cc.sea_level - 0.1, cc.sea_level + 0.1, continent_base)
    continent_base = continent_base + island_mask * ocean_areas * 0.3

    # ----- Step 5: Thresholding and coast smoothing -----
    print("  [continent] Smoothing coastlines...")

    # Create binary land/ocean mask
    binary_mask = (continent_base > cc.sea_level).astype(np.float64)

    # Distance from coast (both directions)
    dist_from_ocean = invert_distance_transform(1 - binary_mask)
    dist_from_land = invert_distance_transform(binary_mask)

    # Normalize distances
    max_dist = max(dist_from_ocean.max(), 1.0)
    dist_from_ocean = dist_from_ocean / max_dist
    dist_from_land = dist_from_land / max_dist

    # Create smooth coast transition
    coast_width = cc.coast_smoothing / res * 10
    smooth_mask = smoothstep(0.0, coast_width, dist_from_ocean)
    smooth_mask = smooth_mask * (1.0 - smoothstep(0.0, coast_width * 0.5, dist_from_land))

    # Final smooth continent mask
    result = blend(
        continent_base * 0.3,  # Ocean values
        continent_base,         # Land values
        smooth_mask
    )

    # Blend with smoothed binary for cleaner result
    blurred_binary = gaussian_blur(binary_mask, sigma=cc.coast_smoothing)
    result = blend(result, blurred_binary, 0.5 * np.ones_like(result))

    result = normalize(result)

    print("  [continent] Done.")
    return result


def get_land_mask(continent_mask: np.ndarray,
                  threshold: float = 0.5) -> np.ndarray:
    """Extract binary land mask from continent mask."""
    return (continent_mask >= threshold).astype(np.float64)


def get_ocean_mask(continent_mask: np.ndarray,
                   threshold: float = 0.5) -> np.ndarray:
    """Extract binary ocean mask from continent mask."""
    return (continent_mask < threshold).astype(np.float64)


def get_coast_mask(continent_mask: np.ndarray,
                   width: float = 0.05) -> np.ndarray:
    """Extract coastal zone mask."""
    threshold = 0.5
    return smoothstep(threshold - width, threshold + width, continent_mask) * \
           (1.0 - smoothstep(threshold + width, threshold + width * 3, continent_mask))
