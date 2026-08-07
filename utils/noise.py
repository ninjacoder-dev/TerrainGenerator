"""
utils/noise.py — Bibliothèque de bruit procédural unifiée.

Implémente : OpenSimplex, fBm, Ridged, Billow, Worley, Domain Warping.
Toutes les fonctions sont vectorisées via numpy pour la performance.
"""

import numpy as np
from typing import Optional, Tuple

try:
    from opensimplex import OpenSimplex
    HAS_OPENSIMPLEX = True
except ImportError:
    HAS_OPENSIMPLEX = False


# ---------------------------------------------------------------------------
# OpenSimplex wrapper
# ---------------------------------------------------------------------------

class SimplexNoise:
    """
    Wrapper autour d'OpenSimplex avec fallback sur un bruit basé gradient.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        if HAS_OPENSIMPLEX:
            self._simplex = OpenSimplex(seed=seed)
        else:
            self._simplex = None
            self._perm = self._build_permutation(seed)

    def _build_permutation(self, seed: int) -> np.ndarray:
        """Build permutation table for fallback noise."""
        rng = np.random.RandomState(seed)
        p = np.arange(256, dtype=np.int32)
        rng.shuffle(p)
        return np.concatenate([p, p])

    def noise2(self, x: float, y: float) -> float:
        """Sample 2D noise at a point. Returns [-1, 1]."""
        if self._simplex is not None:
            return self._simplex.noise2(x, y)
        return self._fallback_noise2(x, y)

    def _fallback_noise2(self, x: float, y: float) -> float:
        """Simple value noise fallback when opensimplex is unavailable."""
        p = self._perm
        xi = int(np.floor(x)) & 255
        yi = int(np.floor(y)) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = xf * xf * (3 - 2 * xf)
        v = yf * yf * (3 - 2 * yf)
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]
        x1 = aa + u * (ba - aa)
        x2 = ab + u * (bb - ab)
        return ((x1 + v * (x2 - x1)) / 128.0) - 1.0

    def noise2_array(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Vectorized 2D noise. xs and ys are 2D coordinate grids.
        Returns 2D array of noise values in [-1, 1].
        """
        if self._simplex is not None:
            # Vectorize with numpy iteration (opensimplex doesn't batch natively)
            result = np.empty(xs.shape, dtype=np.float64)
            flat_x = xs.ravel()
            flat_y = ys.ravel()
            flat_r = result.ravel()
            for i in range(len(flat_x)):
                flat_r[i] = self._simplex.noise2(flat_x[i], flat_y[i])
            return result
        else:
            return self._fallback_array(xs, ys)

    def _fallback_array(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorized fallback noise using numpy."""
        p = self._perm
        xi = np.floor(xs).astype(np.int32) & 255
        yi = np.floor(ys).astype(np.int32) & 255
        xf = xs - np.floor(xs)
        yf = ys - np.floor(ys)
        u = xf * xf * (3.0 - 2.0 * xf)
        v = yf * yf * (3.0 - 2.0 * yf)
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]
        x1 = aa + u * (ba - aa)
        x2 = ab + u * (bb - ab)
        return ((x1 + v * (x2 - x1)) / 128.0) - 1.0


# ---------------------------------------------------------------------------
# Coordinate grid builder
# ---------------------------------------------------------------------------

# Global coordinate offset support for tiled generation
GLOBAL_OFFSET_X: float = 0.0
GLOBAL_OFFSET_Y: float = 0.0


def set_global_offset(x: float, y: float) -> None:
    """Set a global coordinate offset (applied in make_grid).

    Useful for generating tiles from a single continuous noise space
    without generating a single huge heightmap in memory.
    """
    global GLOBAL_OFFSET_X, GLOBAL_OFFSET_Y
    GLOBAL_OFFSET_X = float(x)
    GLOBAL_OFFSET_Y = float(y)


def reset_global_offset() -> None:
    """Reset the global coordinate offset to (0,0)."""
    set_global_offset(0.0, 0.0)


def make_grid(resolution: int, frequency: float = 1.0,
              offset_x: float = 0.0, offset_y: float = 0.0
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D coordinate grid for noise sampling.

    The returned coordinates are in "noise space" and include any
    previously set global offsets so neighbouring tiles are continuous.

    Returns (xs, ys), both shape (resolution, resolution).
    """
    lin = np.linspace(0, 1, resolution, endpoint=False) * frequency
    # Apply both explicit offsets and module-global offsets
    xs, ys = np.meshgrid(lin + offset_x + GLOBAL_OFFSET_X, lin + offset_y + GLOBAL_OFFSET_Y)
    return xs, ys


# ---------------------------------------------------------------------------
# fBm (Fractional Brownian Motion)
# ---------------------------------------------------------------------------

def fbm(noise: SimplexNoise, xs: np.ndarray, ys: np.ndarray,
        octaves: int = 8, persistence: float = 0.5,
        lacunarity: float = 2.0, amplitude: float = 1.0,
        frequency: float = 1.0) -> np.ndarray:
    """
    Standard fractional Brownian motion — layered noise.

    Each octave adds finer detail at lower amplitude.
    """
    result = np.zeros_like(xs)
    amp = amplitude
    freq = frequency
    max_amp = 0.0

    for _ in range(octaves):
        result += amp * noise.noise2_array(xs * freq, ys * freq)
        max_amp += amp
        amp *= persistence
        freq *= lacunarity

    return result / max_amp


# ---------------------------------------------------------------------------
# Ridged Noise
# ---------------------------------------------------------------------------

def ridged(noise: SimplexNoise, xs: np.ndarray, ys: np.ndarray,
           octaves: int = 8, persistence: float = 0.5,
           lacunarity: float = 2.0, amplitude: float = 1.0,
           frequency: float = 1.0, offset: float = 1.0,
           gain: float = 2.0) -> np.ndarray:
    """
    Ridged multifractal noise — creates sharp mountain ridges.

    The absolute value is inverted to create ridges instead of valleys.
    """
    result = np.zeros_like(xs)
    amp = amplitude
    freq = frequency
    weight = 1.0
    max_amp = 0.0

    for _ in range(octaves):
        signal = noise.noise2_array(xs * freq, ys * freq)
        signal = offset - np.abs(signal)
        signal = signal * signal  # Square for sharper ridges
        signal *= weight
        weight = np.clip(signal * gain, 0.0, 1.0) if isinstance(signal, np.ndarray) else max(0, min(1, signal * gain))
        result += amp * signal
        max_amp += amp
        amp *= persistence
        freq *= lacunarity

    return result / max_amp


# ---------------------------------------------------------------------------
# Billow Noise
# ---------------------------------------------------------------------------

def billow(noise: SimplexNoise, xs: np.ndarray, ys: np.ndarray,
           octaves: int = 8, persistence: float = 0.5,
           lacunarity: float = 2.0, amplitude: float = 1.0,
           frequency: float = 1.0) -> np.ndarray:
    """
    Billow noise — rounded, cloud-like formations.

    Similar to fBm but uses abs(noise) for rounded hills.
    """
    result = np.zeros_like(xs)
    amp = amplitude
    freq = frequency
    max_amp = 0.0

    for _ in range(octaves):
        signal = noise.noise2_array(xs * freq, ys * freq)
        signal = 2.0 * np.abs(signal) - 1.0  # Billow transform
        result += amp * signal
        max_amp += amp
        amp *= persistence
        freq *= lacunarity

    return result / max_amp


# ---------------------------------------------------------------------------
# Worley Noise (Cellular)
# ---------------------------------------------------------------------------

def worley(resolution: int, num_points: int = 32, seed: int = 0,
           distance_type: str = "euclidean",
           return_type: str = "F1") -> np.ndarray:
    """
    Worley (cellular) noise — useful for continent masks and organic patterns.

    Parameters:
        resolution: Size of the output grid (resolution × resolution).
        num_points: Number of feature points.
        seed: Random seed.
        distance_type: "euclidean" or "manhattan".
        return_type: "F1" (nearest), "F2" (second nearest), "F2-F1" (edges).
    """
    rng = np.random.RandomState(seed)
    points = rng.rand(num_points, 2) * resolution

    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:resolution, 0:resolution]
    y_coords = y_coords.astype(np.float64)
    x_coords = x_coords.astype(np.float64)

    # Compute distances to all feature points
    distances = np.full((num_points, resolution, resolution), np.inf)
    for i, (px, py) in enumerate(points):
        if distance_type == "manhattan":
            distances[i] = np.abs(x_coords - px) + np.abs(y_coords - py)
        else:
            distances[i] = np.sqrt((x_coords - px) ** 2 + (y_coords - py) ** 2)

    # Sort distances at each pixel
    distances.sort(axis=0)
    f1 = distances[0]
    f2 = distances[1] if num_points > 1 else distances[0]

    if return_type == "F1":
        result = f1
    elif return_type == "F2":
        result = f2
    elif return_type == "F2-F1":
        result = f2 - f1
    else:
        result = f1

    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    return result


# ---------------------------------------------------------------------------
# Domain Warping
# ---------------------------------------------------------------------------

def domain_warp(noise: SimplexNoise, xs: np.ndarray, ys: np.ndarray,
                strength: float = 80.0, warp_octaves: int = 4,
                warp_frequency: float = 0.5, iterations: int = 2,
                noise_func=None, noise_kwargs: Optional[dict] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Domain warping — distort input coordinates using noise.

    Creates organic, natural-looking terrain features.
    Multiple iterations create progressively more complex distortions.

    Returns warped (xs, ys) coordinate pair.
    """
    if noise_func is None:
        noise_func = fbm
    if noise_kwargs is None:
        noise_kwargs = {}

    wx = xs.copy()
    wy = ys.copy()

    for i in range(iterations):
        offset = (i + 1) * 5.2  # Unique offset per iteration
        warp_x = noise_func(
            noise, wx, wy,
            octaves=warp_octaves, frequency=warp_frequency,
            **noise_kwargs
        )
        warp_y = noise_func(
            noise, wx + offset, wy + offset + 1.3,
            octaves=warp_octaves, frequency=warp_frequency,
            **noise_kwargs
        )
        wx = xs + warp_x * strength / xs.shape[0]
        wy = ys + warp_y * strength / ys.shape[1]

    return wx, wy


# ---------------------------------------------------------------------------
# Composite noise stack
# ---------------------------------------------------------------------------

def noise_stack(seed: int, xs: np.ndarray, ys: np.ndarray,
                layers: Optional[list] = None) -> np.ndarray:
    """
    Combine multiple noise layers into a single heightmap.

    Each layer is a dict:
        {"type": "fbm"|"ridged"|"billow", "weight": float, **params}
    """
    if layers is None:
        layers = [
            {"type": "fbm", "weight": 0.4, "octaves": 8, "frequency": 1.0},
            {"type": "ridged", "weight": 0.35, "octaves": 6, "frequency": 1.5},
            {"type": "billow", "weight": 0.15, "octaves": 5, "frequency": 2.0},
            {"type": "fbm", "weight": 0.1, "octaves": 3, "frequency": 4.0},
        ]

    noise_gen = SimplexNoise(seed)
    result = np.zeros_like(xs)
    total_weight = 0.0

    noise_funcs = {
        "fbm": fbm,
        "ridged": ridged,
        "billow": billow,
    }

    for i, layer in enumerate(layers):
        layer_noise = SimplexNoise(seed + i * 1000)
        noise_type = layer.pop("type", "fbm")
        weight = layer.pop("weight", 1.0)
        func = noise_funcs.get(noise_type, fbm)
        result += weight * func(layer_noise, xs, ys, **layer)
        total_weight += weight
        # Restore popped keys for reuse
        layer["type"] = noise_type
        layer["weight"] = weight

    if total_weight > 0:
        result /= total_weight

    return result
