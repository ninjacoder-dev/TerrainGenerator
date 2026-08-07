"""
visualization/lighting.py — Modèles d'éclairage dynamique.
"""

import numpy as np
from typing import Tuple
from exporters.normal import generate_normal_map
from utils.math_utils import clamp


def compute_directional_lighting(heightmap: np.ndarray,
                                 light_dir: Tuple[float, float, float] = (0.5, 0.5, 1.0)
                                 ) -> np.ndarray:
    """Compute diffuse Lambertian shading."""
    normal_map = generate_normal_map(heightmap)
    # Remap normal map back to [-1, 1]
    normals = normal_map * 2.0 - 1.0

    # Normalize light vector
    light = np.array(light_dir, dtype=np.float64)
    light /= np.linalg.norm(light)

    # Dot product
    diffuse = normals[:, :, 0] * light[0] + \
              normals[:, :, 1] * light[1] + \
              normals[:, :, 2] * light[2]

    return clamp(diffuse, 0.0, 1.0)
