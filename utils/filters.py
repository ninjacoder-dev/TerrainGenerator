"""
utils/filters.py — Filtres de traitement d'image pour heightmaps.

Inclut : Gaussian blur, sharpen, median, Sobel edge detection,
highpass, lowpass, unsharp mask.
"""

import numpy as np
from scipy import ndimage
from typing import Optional


def gaussian_blur(heightmap: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian blur to a heightmap.

    Parameters:
        sigma: Standard deviation of the Gaussian kernel.
               Higher = more blur.
    """
    return ndimage.gaussian_filter(heightmap, sigma=sigma)


def sharpen(heightmap: np.ndarray, strength: float = 1.0,
            sigma: float = 2.0) -> np.ndarray:
    """
    Sharpen a heightmap using unsharp mask technique.

    Parameters:
        strength: Sharpening intensity.
        sigma: Blur radius for the unsharp mask.
    """
    blurred = gaussian_blur(heightmap, sigma)
    return heightmap + strength * (heightmap - blurred)


def unsharp_mask(heightmap: np.ndarray, sigma: float = 2.0,
                 amount: float = 1.5, threshold: float = 0.0) -> np.ndarray:
    """
    Professional unsharp mask filter.

    Parameters:
        sigma: Blur radius.
        amount: Sharpening amount (1.0 = subtle, 2.0+ = strong).
        threshold: Minimum difference to sharpen (noise gate).
    """
    blurred = gaussian_blur(heightmap, sigma)
    diff = heightmap - blurred
    if threshold > 0:
        mask = np.abs(diff) > threshold
        diff = diff * mask
    return heightmap + amount * diff


def median_filter(heightmap: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Apply median filter — good for removing noise while preserving edges.

    Parameters:
        size: Filter kernel size (must be odd).
    """
    return ndimage.median_filter(heightmap, size=size)


def sobel_x(heightmap: np.ndarray) -> np.ndarray:
    """Sobel filter in X direction (horizontal edges)."""
    return ndimage.sobel(heightmap, axis=1)


def sobel_y(heightmap: np.ndarray) -> np.ndarray:
    """Sobel filter in Y direction (vertical edges)."""
    return ndimage.sobel(heightmap, axis=0)


def sobel_magnitude(heightmap: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude = sqrt(sobel_x² + sobel_y²)."""
    sx = sobel_x(heightmap)
    sy = sobel_y(heightmap)
    return np.sqrt(sx ** 2 + sy ** 2)


def highpass(heightmap: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    High-pass filter — extracts fine details by removing low frequencies.
    """
    return heightmap - gaussian_blur(heightmap, sigma)


def lowpass(heightmap: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Low-pass filter — removes fine details, keeps broad shapes.
    """
    return gaussian_blur(heightmap, sigma)


def bilateral_filter(heightmap: np.ndarray, sigma_spatial: float = 3.0,
                     sigma_range: float = 0.1, size: int = 5) -> np.ndarray:
    """
    Approximate bilateral filter — smooths while preserving edges.
    Uses a separable approximation for performance.
    """
    result = np.zeros_like(heightmap)
    half = size // 2
    padded = np.pad(heightmap, half, mode='reflect')

    # Gaussian spatial weights
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    spatial_weights = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_spatial ** 2))

    h, w = heightmap.shape
    for i in range(h):
        for j in range(w):
            patch = padded[i:i + size, j:j + size]
            center_val = heightmap[i, j]
            range_weights = np.exp(-((patch - center_val) ** 2) / (2 * sigma_range ** 2))
            weights = spatial_weights * range_weights
            total = weights.sum()
            if total > 0:
                result[i, j] = (patch * weights).sum() / total
            else:
                result[i, j] = center_val

    return result


def erode_filter(heightmap: np.ndarray, size: int = 3) -> np.ndarray:
    """Morphological erosion — minimum in neighborhood."""
    return ndimage.minimum_filter(heightmap, size=size)


def dilate_filter(heightmap: np.ndarray, size: int = 3) -> np.ndarray:
    """Morphological dilation — maximum in neighborhood."""
    return ndimage.maximum_filter(heightmap, size=size)


def contrast_enhance(heightmap: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Enhance contrast around the mean."""
    mean = heightmap.mean()
    return mean + (heightmap - mean) * factor


def clamp_filter(heightmap: np.ndarray, lo: float = 0.0,
                 hi: float = 1.0) -> np.ndarray:
    """Clamp values to a range."""
    return np.clip(heightmap, lo, hi)
