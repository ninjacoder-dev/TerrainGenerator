"""
utils/interpolation.py — Interpolation et rescaling de heightmaps.

Inclut : bilinéaire, bicubique, upscale/downscale.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple


def bilinear_sample(heightmap: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation sampling of a heightmap at floating-point coordinates.

    Parameters:
        heightmap: 2D array
        x, y: Arrays of floating-point coordinates (can be any shape)

    Returns:
        Interpolated values at the given coordinates, same shape as x/y.
    """
    h, w = heightmap.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    fx = x - x0
    fy = y - y0

    val00 = heightmap[y0, x0]
    val10 = heightmap[y0, x1]
    val01 = heightmap[y1, x0]
    val11 = heightmap[y1, x1]

    top = val00 * (1 - fx) + val10 * fx
    bottom = val01 * (1 - fx) + val11 * fx
    return top * (1 - fy) + bottom * fy


def bicubic_sample(heightmap: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bicubic interpolation sampling using scipy's map_coordinates.
    """
    coords = np.array([y.ravel(), x.ravel()])
    result = ndimage.map_coordinates(heightmap, coords, order=3, mode='reflect')
    return result.reshape(x.shape)


def rescale(heightmap: np.ndarray, new_size: int,
            method: str = "bicubic") -> np.ndarray:
    """
    Rescale a heightmap to a new resolution.

    Parameters:
        heightmap: 2D input array
        new_size: Target resolution (new_size × new_size)
        method: "nearest", "bilinear", or "bicubic"
    """
    order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
    order = order_map.get(method, 3)

    zoom_y = new_size / heightmap.shape[0]
    zoom_x = new_size / heightmap.shape[1]

    return ndimage.zoom(heightmap, (zoom_y, zoom_x), order=order)


def upscale(heightmap: np.ndarray, factor: int = 2,
            method: str = "bicubic") -> np.ndarray:
    """Upscale heightmap by an integer factor."""
    new_h = heightmap.shape[0] * factor
    new_w = heightmap.shape[1] * factor
    return rescale(heightmap, max(new_h, new_w), method)


def downscale(heightmap: np.ndarray, factor: int = 2,
              method: str = "bicubic") -> np.ndarray:
    """Downscale heightmap by an integer factor."""
    new_h = max(1, heightmap.shape[0] // factor)
    new_w = max(1, heightmap.shape[1] // factor)
    return rescale(heightmap, max(new_h, new_w), method)


def tile_seamless(heightmap: np.ndarray) -> np.ndarray:
    """
    Make a heightmap tile-seamlessly by blending edges.
    Uses mirrored copies to create smooth transitions.
    """
    h, w = heightmap.shape
    result = heightmap.copy()
    blend_size = min(h, w) // 8

    if blend_size < 2:
        return result

    # Horizontal blending
    for i in range(blend_size):
        t = i / blend_size
        t = t * t * (3 - 2 * t)  # smoothstep
        result[:, i] = heightmap[:, i] * t + heightmap[:, w - blend_size + i] * (1 - t)
        result[:, w - blend_size + i] = heightmap[:, w - blend_size + i] * t + heightmap[:, i] * (1 - t)

    # Vertical blending
    for i in range(blend_size):
        t = i / blend_size
        t = t * t * (3 - 2 * t)
        result[i, :] = result[i, :] * t + result[h - blend_size + i, :] * (1 - t)
        result[h - blend_size + i, :] = result[h - blend_size + i, :] * t + result[i, :] * (1 - t)

    return result
