"""
exporters/splatmap.py — Export Splat map RGBA.

Canaux :
  R: Rock / Cliff
  G: Grass / Forest
  B: Sand / Desert
  A: Snow / Mud
"""

import numpy as np
from PIL import Image
from pathlib import Path


def generate_splatmap(rock_mask: np.ndarray,
                      grass_mask: np.ndarray,
                      sand_mask: np.ndarray,
                      snow_mask: np.ndarray) -> np.ndarray:
    """Generate RGBA splatmap array in [0, 255]."""
    r = (np.clip(rock_mask, 0.0, 1.0) * 255).astype(np.uint8)
    g = (np.clip(grass_mask, 0.0, 1.0) * 255).astype(np.uint8)
    b = (np.clip(sand_mask, 0.0, 1.0) * 255).astype(np.uint8)
    a = (np.clip(snow_mask, 0.0, 1.0) * 255).astype(np.uint8)

    splat_rgba = np.dstack((r, g, b, a))
    return splat_rgba


def export_splatmap(rock_mask: np.ndarray,
                    grass_mask: np.ndarray,
                    sand_mask: np.ndarray,
                    snow_mask: np.ndarray,
                    filepath: str) -> str:
    """Export 4-channel RGBA texture splatmap PNG."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    splat = generate_splatmap(rock_mask, grass_mask, sand_mask, snow_mask)

    img = Image.fromarray(splat, mode='RGBA')
    img.save(filepath)
    print(f"  [export] Saved Splatmap RGBA: {filepath}")
    return filepath
