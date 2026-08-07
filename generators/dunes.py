"""
generators/dunes.py — Générateur de dunes (simulation éolienne).

Création :
  - Dunes linéaires
  - Barkhanes (crescent dunes)
  - Mégadunes
"""

import numpy as np

from config import WorldConfig
from utils.noise import SimplexNoise, make_grid, fbm
from utils.math_utils import normalize, smoothstep, clamp


def generate_dunes(config: WorldConfig,
                   heightmap: np.ndarray) -> np.ndarray:
    """
    Add sand dune fields aligned with wind direction.
    """
    res = config.resolution
    dc = config.dune
    seed = config.sub_seed("dunes")

    print("  [dunes] Generating sand dunes...")
    result = heightmap.copy()

    # Dunes occur primarily in arid lowland basins
    dune_area_mask = (heightmap > config.sea_level + 0.01) & (heightmap < 0.45)
    dune_weight = smoothstep(config.sea_level + 0.02, config.sea_level + 0.08, heightmap) * \
                  (1.0 - smoothstep(0.35, 0.48, heightmap))

    rad = np.radians(dc.wind_direction)
    cos_w, sin_w = np.cos(rad), np.sin(rad)

    lin = np.linspace(0, 1, res, endpoint=False)
    x_grid, y_grid = np.meshgrid(lin, lin)

    # Rotate coordinates aligned with wind
    u = x_grid * cos_w + y_grid * sin_w
    v = -x_grid * sin_w + y_grid * cos_w

    freq = 1.0 / max(dc.dune_spacing, 0.001)

    # Asymmetric transverse dune profile (slip face)
    wave = np.sin(u * freq * 2 * np.pi)
    asymmetric_wave = np.where(wave > 0, wave ** 0.5, -((-wave) ** 1.5))
    dunes = (asymmetric_wave * 0.5 + 0.5)

    # Add megadune variation using noise
    noise = SimplexNoise(seed)
    xs, ys = make_grid(res, frequency=2.0)
    dune_noise = fbm(noise, xs, ys, octaves=4, frequency=2.0, persistence=0.5)
    dunes = dunes * (0.7 + 0.3 * dune_noise)

    dune_field = dunes * dc.dune_height * dune_weight
    result += dune_field

    result = clamp(result, 0.0, 1.0)
    print("  [dunes] Done.")
    return result
