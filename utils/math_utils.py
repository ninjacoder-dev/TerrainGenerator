"""
utils/math_utils.py — Fonctions mathématiques utilitaires pour le terrain.

Inclut : normalize, clamp, lerp, smoothstep, gradient, Laplacien, courbure,
distance transform, et opérations sur heightmaps.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Scalar operations
# ---------------------------------------------------------------------------

def normalize(data: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Normalize array to [vmin, vmax] range."""
    d_min = data.min()
    d_max = data.max()
    if d_max - d_min < 1e-10:
        return np.full_like(data, (vmin + vmax) / 2.0)
    return vmin + (data - d_min) * (vmax - vmin) / (d_max - d_min)


def clamp(data: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Clamp array values to [lo, hi]."""
    return np.clip(data, lo, hi)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between a and b by factor t."""
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep interpolation."""
    t = clamp((x - edge0) / (edge1 - edge0 + 1e-10), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Ken Perlin's smootherstep — C2-continuous."""
    t = clamp((x - edge0) / (edge1 - edge0 + 1e-10), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def remap(value: np.ndarray,
          in_min: float, in_max: float,
          out_min: float, out_max: float) -> np.ndarray:
    """Remap value from [in_min, in_max] to [out_min, out_max]."""
    t = (value - in_min) / (in_max - in_min + 1e-10)
    return out_min + t * (out_max - out_min)


def power_curve(data: np.ndarray, exponent: float) -> np.ndarray:
    """Apply power curve — values must be in [0, 1]."""
    return np.power(clamp(data, 0.0, 1.0), exponent)


# ---------------------------------------------------------------------------
# Heightmap derivatives
# ---------------------------------------------------------------------------

def gradient_map(heightmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient (partial derivatives) of heightmap.
    Returns (dy, dx) — the vertical and horizontal gradients.
    """
    dy, dx = np.gradient(heightmap)
    return dy, dx


def slope_map(heightmap: np.ndarray) -> np.ndarray:
    """
    Compute slope magnitude from heightmap.
    slope = sqrt(dx² + dy²)
    """
    dy, dx = np.gradient(heightmap)
    return np.sqrt(dx ** 2 + dy ** 2)


def slope_angle(heightmap: np.ndarray, world_size: float = 1.0) -> np.ndarray:
    """
    Compute slope angle in degrees.
    Accounts for world scale.
    """
    dy, dx = np.gradient(heightmap)
    scale = heightmap.shape[0] / world_size
    return np.degrees(np.arctan(np.sqrt((dx * scale) ** 2 + (dy * scale) ** 2)))


def aspect_map(heightmap: np.ndarray) -> np.ndarray:
    """
    Compute aspect (direction of steepest slope) in radians.
    0 = North, π/2 = East, etc.
    """
    dy, dx = np.gradient(heightmap)
    return np.arctan2(-dx, dy)


def laplacian(heightmap: np.ndarray) -> np.ndarray:
    """
    Compute Laplacian (second derivative sum).
    Positive = concave (valley), Negative = convex (ridge).
    """
    return ndimage.laplace(heightmap)


def curvature_map(heightmap: np.ndarray) -> np.ndarray:
    """
    Compute mean curvature.
    Positive = concave (accumulation), Negative = convex (dispersion).
    """
    dy, dx = np.gradient(heightmap)
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)
    return (dxx + dyy) / 2.0


def profile_curvature(heightmap: np.ndarray) -> np.ndarray:
    """Profile curvature — curvature in the direction of steepest slope."""
    dy, dx = np.gradient(heightmap)
    dyy, _ = np.gradient(dy)
    _, dxx = np.gradient(dx)
    dxy, _ = np.gradient(dx)

    p = dx ** 2 + dy ** 2
    mask = p > 1e-10
    result = np.zeros_like(heightmap)
    result[mask] = (
        dx[mask] ** 2 * dxx[mask]
        + 2.0 * dx[mask] * dy[mask] * dxy[mask]
        + dy[mask] ** 2 * dyy[mask]
    ) / (p[mask] * np.sqrt(p[mask] + 1.0))
    return result


def plan_curvature(heightmap: np.ndarray) -> np.ndarray:
    """Plan curvature — curvature perpendicular to the slope direction."""
    dy, dx = np.gradient(heightmap)
    dyy, _ = np.gradient(dy)
    _, dxx = np.gradient(dx)
    dxy, _ = np.gradient(dx)

    p = dx ** 2 + dy ** 2
    mask = p > 1e-10
    result = np.zeros_like(heightmap)
    result[mask] = (
        dx[mask] ** 2 * dyy[mask]
        - 2.0 * dx[mask] * dy[mask] * dxy[mask]
        + dy[mask] ** 2 * dxx[mask]
    ) / (p[mask] ** 1.5 + 1e-10)
    return result


# ---------------------------------------------------------------------------
# Heightmap operations
# ---------------------------------------------------------------------------

def blend(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend two heightmaps using a mask (0 = a, 1 = b)."""
    mask_c = clamp(mask, 0.0, 1.0)
    return a * (1.0 - mask_c) + b * mask_c


def terrace(heightmap: np.ndarray, levels: int = 8) -> np.ndarray:
    """Create terraced/stepped terrain."""
    return np.floor(heightmap * levels) / levels


def radial_gradient(resolution: int, center: Optional[Tuple[float, float]] = None,
                    radius: float = 0.5) -> np.ndarray:
    """Create a radial gradient mask (1.0 at center, 0.0 at radius)."""
    if center is None:
        center = (0.5, 0.5)
    y, x = np.mgrid[0:resolution, 0:resolution] / resolution
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return clamp(1.0 - dist / radius, 0.0, 1.0)


def distance_transform(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance transform.
    Returns distance from each pixel to the nearest True (nonzero) pixel.
    """
    return ndimage.distance_transform_edt(1 - binary_mask.astype(np.float64))


def invert_distance_transform(binary_mask: np.ndarray) -> np.ndarray:
    """Distance from each pixel to the nearest False (zero) pixel."""
    return ndimage.distance_transform_edt(binary_mask.astype(np.float64))


def apply_sea_level(heightmap: np.ndarray, sea_level: float) -> np.ndarray:
    """Flatten everything below sea_level to sea_level."""
    result = heightmap.copy()
    result[result < sea_level] = sea_level
    return result


def flow_direction(heightmap: np.ndarray) -> np.ndarray:
    """
    Compute flow direction for each cell (D8 algorithm).
    Returns an array where each cell contains the index (0-7) of the
    steepest downhill neighbor.

    Directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
    """
    h, w = heightmap.shape
    flow = np.full((h, w), -1, dtype=np.int32)
    max_drop = np.full((h, w), -np.inf, dtype=np.float64)

    # Neighbor offsets (dy, dx) for 8 directions
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]
    distances = [1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414]

    for direction, ((dy, dx), dist) in enumerate(zip(offsets, distances)):
        # Compute slope to this neighbor
        y_from = max(0, -dy)
        y_to = h - max(0, dy)
        x_from = max(0, -dx)
        x_to = w - max(0, dx)

        center = heightmap[y_from:y_to, x_from:x_to]
        neighbor = heightmap[y_from + dy:y_to + dy, x_from + dx:x_to + dx]

        drop = (center - neighbor) / dist
        
        # Get the corresponding region in flow and max_drop
        flow_region = flow[y_from:y_to, x_from:x_to]
        max_drop_region = max_drop[y_from:y_to, x_from:x_to]

        # Update where this direction has steeper drop
        mask = drop > max_drop_region
        flow_region[mask] = direction
        max_drop_region[mask] = drop[mask]

    return flow


def flow_accumulation(heightmap: np.ndarray) -> np.ndarray:
    """
    Compute flow accumulation (how many cells drain through each cell).
    Higher values indicate rivers and drainage paths.
    """
    h, w = heightmap.shape
    accumulation = np.ones((h, w), dtype=np.float64)

    # Sort cells by height (highest first)
    flat_indices = np.argsort(-heightmap.ravel())

    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]
    distances = [1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414]

    for idx in flat_indices:
        y, x = divmod(idx, w)
        current_h = heightmap[y, x]

        # Find steepest downhill neighbor
        best_dir = -1
        best_drop = 0.0
        for d, ((dy, dx), dist) in enumerate(zip(offsets, distances)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                drop = (current_h - heightmap[ny, nx]) / dist
                if drop > best_drop:
                    best_drop = drop
                    best_dir = d

        if best_dir >= 0:
            dy, dx = offsets[best_dir]
            accumulation[y + dy, x + dx] += accumulation[y, x]

    return accumulation
