"""
utils/random.py — Générateur aléatoire déterministe et distributions spatiales.

Inclut : Random seedé, Poisson disk sampling, distributions spatiales.
"""

import numpy as np
from typing import List, Tuple, Optional


class SeededRandom:
    """
    Générateur aléatoire déterministe.
    Wrapper autour de numpy.random.RandomState pour garantir la reproductibilité.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def random(self) -> float:
        """Uniform random float in [0, 1)."""
        return self._rng.random()

    def uniform(self, lo: float = 0.0, hi: float = 1.0,
                size=None) -> np.ndarray:
        """Uniform random in [lo, hi)."""
        return self._rng.uniform(lo, hi, size)

    def randint(self, lo: int, hi: int, size=None):
        """Random integer in [lo, hi)."""
        return self._rng.randint(lo, hi, size)

    def normal(self, mean: float = 0.0, std: float = 1.0,
               size=None) -> np.ndarray:
        """Normal distribution."""
        return self._rng.normal(mean, std, size)

    def choice(self, arr, size=None, replace: bool = True, p=None):
        """Random choice from array."""
        return self._rng.choice(arr, size=size, replace=replace, p=p)

    def shuffle(self, arr: np.ndarray) -> None:
        """In-place shuffle."""
        self._rng.shuffle(arr)

    def rand_point(self, bounds: Tuple[float, float, float, float]
                   ) -> Tuple[float, float]:
        """Random point within (x_min, y_min, x_max, y_max) bounds."""
        x = self._rng.uniform(bounds[0], bounds[2])
        y = self._rng.uniform(bounds[1], bounds[3])
        return (x, y)

    @property
    def rng(self) -> np.random.RandomState:
        """Access underlying RandomState for compatibility."""
        return self._rng


def poisson_disk_sampling(width: float, height: float,
                          min_distance: float, seed: int = 0,
                          max_attempts: int = 30
                          ) -> List[Tuple[float, float]]:
    """
    Poisson disk sampling — generates evenly-spaced random points.

    Uses Bridson's algorithm for O(n) performance.

    Parameters:
        width, height: Domain size.
        min_distance: Minimum distance between any two points.
        seed: Random seed.
        max_attempts: Attempts per active point before giving up.

    Returns:
        List of (x, y) tuples.
    """
    rng = np.random.RandomState(seed)
    cell_size = min_distance / np.sqrt(2)
    grid_w = int(np.ceil(width / cell_size))
    grid_h = int(np.ceil(height / cell_size))

    grid = {}  # (gx, gy) -> point index
    points: List[Tuple[float, float]] = []
    active: List[int] = []

    def grid_coords(x: float, y: float) -> Tuple[int, int]:
        return int(x / cell_size), int(y / cell_size)

    def is_valid(x: float, y: float) -> bool:
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        gx, gy = grid_coords(x, y)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                key = (gx + dx, gy + dy)
                if key in grid:
                    ox, oy = points[grid[key]]
                    dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                    if dist < min_distance:
                        return False
        return True

    # Initial point
    x0 = rng.uniform(0, width)
    y0 = rng.uniform(0, height)
    points.append((x0, y0))
    active.append(0)
    grid[grid_coords(x0, y0)] = 0

    while active:
        idx = rng.randint(0, len(active))
        px, py = points[active[idx]]
        found = False

        for _ in range(max_attempts):
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(min_distance, 2 * min_distance)
            nx = px + dist * np.cos(angle)
            ny = py + dist * np.sin(angle)

            if is_valid(nx, ny):
                new_idx = len(points)
                points.append((nx, ny))
                active.append(new_idx)
                grid[grid_coords(nx, ny)] = new_idx
                found = True
                break

        if not found:
            active.pop(idx)

    return points


def scatter_points(mask: np.ndarray, density: float = 0.01,
                   seed: int = 0, min_distance: float = 0.0
                   ) -> List[Tuple[int, int]]:
    """
    Scatter random points on a mask where mask > 0.

    Parameters:
        mask: 2D array (0 = empty, >0 = allow placement).
        density: Probability of placing a point at each valid cell.
        seed: Random seed.
        min_distance: Minimum distance between points (0 = no constraint).

    Returns:
        List of (row, col) tuples.
    """
    rng = np.random.RandomState(seed)
    valid = np.argwhere(mask > 0)

    if len(valid) == 0:
        return []

    # Random subset based on density
    num_points = max(1, int(len(valid) * density))
    indices = rng.choice(len(valid), size=min(num_points, len(valid)), replace=False)
    selected = valid[indices]

    if min_distance <= 0:
        return [(int(r), int(c)) for r, c in selected]

    # Filter by minimum distance
    result = []
    for r, c in selected:
        too_close = False
        for pr, pc in result:
            if np.sqrt((r - pr) ** 2 + (c - pc) ** 2) < min_distance:
                too_close = True
                break
        if not too_close:
            result.append((int(r), int(c)))

    return result


def jittered_grid(width: int, height: int, spacing: int,
                  jitter: float = 0.4, seed: int = 0
                  ) -> List[Tuple[float, float]]:
    """
    Create a jittered grid of points — regular grid with random offsets.

    Parameters:
        width, height: Domain size.
        spacing: Base grid spacing.
        jitter: Maximum offset as fraction of spacing (0 = regular grid).
        seed: Random seed.

    Returns:
        List of (x, y) tuples.
    """
    rng = np.random.RandomState(seed)
    points = []

    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            jx = x + rng.uniform(-jitter, jitter) * spacing
            jy = y + rng.uniform(-jitter, jitter) * spacing
            jx = np.clip(jx, 0, width - 1)
            jy = np.clip(jy, 0, height - 1)
            points.append((jx, jy))

    return points
