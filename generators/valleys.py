"""
generators/valleys.py — Générateur de vallées.

Types :
  - Vallées glaciaires (profil en U)
  - Vallées fluviales (profil en V)
  - Vallées tectoniques (rifts, grabens)
"""

import numpy as np
from typing import Optional

from config import WorldConfig
from utils.noise import SimplexNoise, make_grid, fbm, domain_warp
from utils.math_utils import normalize, smoothstep, clamp, slope_map
from utils.filters import gaussian_blur


def generate_valleys(config: WorldConfig,
                     heightmap: np.ndarray) -> np.ndarray:
    """
    Carve valleys into the heightmap.
    """
    res = config.resolution
    vc = config.valley
    seed = config.sub_seed("valleys")

    print("  [valleys] Generating valleys...")
    result = heightmap.copy()

    # ----- Glacial valleys (U-shape) -----
    print("  [valleys] Carving glacial valleys...")
    glacial = _generate_glacial_valleys(res, seed, vc.glacial_width)
    result -= glacial * vc.glacial_width * smoothstep(0.5, 0.8, heightmap)

    # ----- Fluvial valleys (V-shape) -----
    print("  [valleys] Carving fluvial valleys...")
    fluvial = _generate_fluvial_valleys(res, seed + 100, vc.fluvial_depth)
    result -= fluvial * vc.fluvial_depth * smoothstep(0.3, 0.6, heightmap)

    # ----- Tectonic valleys (rift) -----
    print("  [valleys] Carving tectonic valleys...")
    tectonic = _generate_tectonic_valleys(res, seed + 200, vc.tectonic_scale)
    result -= tectonic * vc.tectonic_scale * 0.3

    result = clamp(result, 0.0, 1.0)
    print("  [valleys] Done.")
    return result


def _generate_glacial_valleys(res: int, seed: int,
                              width: float) -> np.ndarray:
    """
    Glacial valley: wide, flat-bottomed U-shape.
    Created by carving smooth channels through terrain.
    """
    noise = SimplexNoise(seed)
    xs, ys = make_grid(res, frequency=1.5)

    # Domain warp for sinuous paths
    warp_noise = SimplexNoise(seed + 10)
    wxs, wys = domain_warp(warp_noise, xs, ys, strength=50.0,
                           warp_octaves=3, warp_frequency=0.8, iterations=2)

    # Create valley channels using stretched noise
    valley_path = fbm(noise, wxs * 3.0, wys * 0.5,
                      octaves=4, frequency=1.5, persistence=0.5)
    valley_path = normalize(valley_path)

    # U-shape profile: flat bottom + steep walls
    valley_mask = smoothstep(0.4, 0.5, valley_path) * (1.0 - smoothstep(0.5, 0.6, valley_path))

    # Widen with blur for U-shape
    valley_mask = gaussian_blur(valley_mask, sigma=res / 150)
    return valley_mask


def _generate_fluvial_valleys(res: int, seed: int,
                              depth: float) -> np.ndarray:
    """
    Fluvial valley: narrow, V-shaped, follows water flow.
    """
    noise = SimplexNoise(seed)
    xs, ys = make_grid(res, frequency=2.0)

    # Sinuous channel
    warp_noise = SimplexNoise(seed + 10)
    wxs, wys = domain_warp(warp_noise, xs, ys, strength=30.0,
                           warp_octaves=4, warp_frequency=1.0, iterations=2)

    channel = fbm(noise, wxs * 4.0, wys * 0.8,
                  octaves=5, frequency=2.0, persistence=0.5)
    channel = normalize(channel)

    # V-shape: narrow, sharp
    valley_mask = 1.0 - np.abs(channel - 0.5) * 2.0
    valley_mask = np.power(clamp(valley_mask, 0.0, 1.0), 3.0)  # Sharp V

    # Less blur for V-shape
    valley_mask = gaussian_blur(valley_mask, sigma=res / 400)
    return valley_mask


def _generate_tectonic_valleys(res: int, seed: int,
                               scale: float) -> np.ndarray:
    """
    Tectonic valley: straight, wide rift/graben with steep walls.
    """
    noise = SimplexNoise(seed)
    xs, ys = make_grid(res, frequency=0.5)

    # Long, oriented features
    rift = fbm(noise, xs * 0.5, ys * 2.0,
               octaves=3, frequency=0.5, persistence=0.4)
    rift = normalize(rift)

    # Create rift zones
    rift_mask = smoothstep(0.45, 0.5, rift) * (1.0 - smoothstep(0.5, 0.55, rift))

    # Steep walls
    rift_mask = gaussian_blur(rift_mask, sigma=res / 300)
    return rift_mask
