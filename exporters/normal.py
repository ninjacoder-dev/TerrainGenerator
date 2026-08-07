"""
exporters/normal.py — Génération et export de Normal Map.
"""

import numpy as np
from PIL import Image
from pathlib import Path


def generate_normal_map(heightmap: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Compute normal vectors (RGB) from heightmap gradient.
    Returns (H, W, 3) float array in [0.0, 1.0].
    """
    dy, dx = np.gradient(heightmap)

    # Tangent / Bitangent cross product for normal vector
    nz = 1.0 / max(strength, 0.01)
    norm = np.sqrt(dx ** 2 + dy ** 2 + nz ** 2)

    nx = -dx / norm
    ny = -dy / norm
    nz = nz / norm

    # Remap from [-1, 1] to [0, 1]
    normal_rgb = np.dstack(((nx + 1.0) * 0.5, (ny + 1.0) * 0.5, (nz + 1.0) * 0.5))
    return normal_rgb


def export_normal_map(heightmap: np.ndarray, filepath: str, strength: float = 2.0) -> str:
    """Export normal map as 8-bit RGB PNG."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    normal_rgb = generate_normal_map(heightmap, strength)

    uint8_rgb = (np.clip(normal_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(uint8_rgb, mode='RGB')
    img.save(filepath)
    print(f"  [export] Saved Normal Map: {filepath}")
    return filepath
