"""
utils/threading.py — Abstraction GPU/CPU et parallélisation.

Fournit :
- Détection et fallback GPU (CuPy) → CPU (NumPy)
- Wrapper multiprocessing pour traitement par tuiles
- Accélération Numba pour boucles critiques
"""

import numpy as np
import multiprocessing as mp
from typing import Callable, Any, List, Tuple, Optional
from functools import wraps
import time
import sys


# ---------------------------------------------------------------------------
# GPU / CPU abstraction
# ---------------------------------------------------------------------------

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Provide no-op decorators when numba is not available
    def njit(*args, **kwargs):
        """Fallback no-op decorator when numba is unavailable."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

    prange = range


def get_array_module(prefer_gpu: bool = True):
    """
    Return the appropriate array module (cupy or numpy).

    Usage:
        xp = get_array_module()
        arr = xp.zeros((1024, 1024))
    """
    if prefer_gpu and HAS_GPU:
        return cp
    return np


def to_gpu(array: np.ndarray) -> Any:
    """Transfer numpy array to GPU (no-op if GPU unavailable)."""
    if HAS_GPU:
        return cp.asarray(array)
    return array


def to_cpu(array: Any) -> np.ndarray:
    """Transfer GPU array to CPU (no-op if already numpy)."""
    if HAS_GPU and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return HAS_GPU


def numba_available() -> bool:
    """Check if Numba JIT is available."""
    return HAS_NUMBA


# ---------------------------------------------------------------------------
# Tile-based multiprocessing
# ---------------------------------------------------------------------------

def split_into_tiles(heightmap: np.ndarray, tile_size: int = 256,
                     overlap: int = 16
                     ) -> List[Tuple[int, int, np.ndarray]]:
    """
    Split a heightmap into overlapping tiles for parallel processing.

    Returns list of (row_start, col_start, tile_data).
    """
    h, w = heightmap.shape
    tiles = []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_start = max(0, y - overlap)
            x_start = max(0, x - overlap)
            y_end = min(h, y + tile_size + overlap)
            x_end = min(w, x + tile_size + overlap)

            tile = heightmap[y_start:y_end, x_start:x_end].copy()
            tiles.append((y, x, y_start, x_start, y_end, x_end, tile))

    return tiles


def stitch_tiles(tiles: List[Tuple], output_shape: Tuple[int, int],
                 tile_size: int = 256, overlap: int = 16
                 ) -> np.ndarray:
    """
    Stitch processed tiles back together, blending the overlap regions.
    """
    result = np.zeros(output_shape, dtype=np.float64)
    weight = np.zeros(output_shape, dtype=np.float64)

    for (y, x, y_start, x_start, y_end, x_end, tile) in tiles:
        # Create a weight mask that fades in the overlap region
        tile_h, tile_w = tile.shape
        w_mask = np.ones((tile_h, tile_w), dtype=np.float64)

        # Fade borders
        fade = min(overlap, tile_h // 4, tile_w // 4)
        if fade > 0:
            for i in range(fade):
                t = (i + 1) / (fade + 1)
                if y_start < y:
                    w_mask[i, :] *= t
                if x_start < x:
                    w_mask[:, i] *= t
                if y_end > y + tile_size:
                    w_mask[-(i + 1), :] *= t
                if x_end > x + tile_size:
                    w_mask[:, -(i + 1)] *= t

        result[y_start:y_end, x_start:x_end] += tile * w_mask
        weight[y_start:y_end, x_start:x_end] += w_mask

    mask = weight > 0
    result[mask] /= weight[mask]
    return result


def process_tiles_parallel(heightmap: np.ndarray,
                           process_func: Callable,
                           tile_size: int = 256,
                           overlap: int = 16,
                           num_workers: Optional[int] = None,
                           **kwargs) -> np.ndarray:
    """
    Process a heightmap in parallel using tiles.

    Parameters:
        heightmap: Input 2D array.
        process_func: Function(tile_data, **kwargs) -> processed_tile.
        tile_size: Size of each processing tile.
        overlap: Overlap between tiles for seamless stitching.
        num_workers: Number of parallel workers (default: CPU count).
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    tiles = split_into_tiles(heightmap, tile_size, overlap)

    # Process tiles (serial for now, can be parallelized with Pool)
    processed = []
    for (y, x, y_start, x_start, y_end, x_end, tile) in tiles:
        result_tile = process_func(tile, **kwargs)
        processed.append((y, x, y_start, x_start, y_end, x_end, result_tile))

    return stitch_tiles(processed, heightmap.shape, tile_size, overlap)


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Simple progress bar for long-running operations."""

    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, n: int = 1) -> None:
        self.current += n
        self._print()

    def _print(self) -> None:
        if self.total <= 0:
            return
        pct = self.current / self.total * 100
        elapsed = time.time() - self.start_time
        bar_len = 40
        filled = int(bar_len * self.current / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)

        eta = ""
        if self.current > 0:
            remaining = elapsed / self.current * (self.total - self.current)
            eta = f" ETA: {remaining:.0f}s"

        sys.stdout.write(
            f"\r  {self.description} [{bar}] {pct:.1f}%{eta}  "
        )
        sys.stdout.flush()

        if self.current >= self.total:
            elapsed_total = time.time() - self.start_time
            sys.stdout.write(f"\r  {self.description} [{bar}] 100.0% — {elapsed_total:.1f}s\n")
            sys.stdout.flush()

    def done(self) -> None:
        self.current = self.total
        self._print()
