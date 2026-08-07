"""
generators/tectonic.py — Simulation de plaques tectoniques.

Chaque plaque possède :
  - une vitesse
  - une direction
  - un type (continentale / océanique)

Les interactions entre plaques produisent :
  - montagnes (convergence continent-continent)
  - failles (décrochement)
  - volcans (subduction)
  - fosses océaniques (subduction océan-océan)
"""

import numpy as np
from scipy import ndimage
from typing import List, Tuple, Optional

from config import WorldConfig
from utils.noise import SimplexNoise, make_grid, fbm
from utils.math_utils import normalize, smoothstep, clamp
from utils.random import SeededRandom


class TectonicPlate:
    """Représentation d'une plaque tectonique."""

    def __init__(self, center: Tuple[float, float],
                 velocity: Tuple[float, float],
                 plate_type: str = "continental",
                 plate_id: int = 0):
        self.center = center
        self.velocity = velocity  # (vx, vy) direction + vitesse
        self.plate_type = plate_type  # "continental" ou "oceanic"
        self.plate_id = plate_id

    @property
    def speed(self) -> float:
        return np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)

    @property
    def direction(self) -> float:
        """Direction en radians."""
        return np.arctan2(self.velocity[1], self.velocity[0])


def generate_tectonic_plates(config: WorldConfig
                             ) -> Tuple[np.ndarray, np.ndarray, List[TectonicPlate]]:
    """
    Generate tectonic plate boundaries and interaction zones.

    Returns:
        plate_map: 2D array of plate IDs
        boundary_map: 2D array of boundary intensities (0-1)
        plates: List of TectonicPlate objects
    """
    res = config.resolution
    tc = config.tectonic
    seed = config.sub_seed("tectonic")
    rng = SeededRandom(seed)

    print(f"  [tectonic] Generating {tc.num_plates} tectonic plates...")

    # ----- Generate plate centers (Voronoi) -----
    plates = []
    for i in range(tc.num_plates):
        cx = rng.uniform(0.05, 0.95)
        cy = rng.uniform(0.05, 0.95)
        vx = rng.uniform(-1.0, 1.0)
        vy = rng.uniform(-1.0, 1.0)
        # Normalize velocity
        speed = rng.uniform(0.3, 1.0)
        mag = np.sqrt(vx ** 2 + vy ** 2) + 1e-10
        vx = vx / mag * speed
        vy = vy / mag * speed

        plate_type = "continental" if rng.random() > 0.4 else "oceanic"
        plates.append(TectonicPlate(
            center=(cx, cy),
            velocity=(vx, vy),
            plate_type=plate_type,
            plate_id=i
        ))

    # ----- Create Voronoi plate map with perturbation -----
    print("  [tectonic] Computing Voronoi plate regions...")
    y_coords, x_coords = np.mgrid[0:res, 0:res] / res

    # Add noise perturbation for organic boundaries
    noise = SimplexNoise(seed + 100)
    xs, ys = make_grid(res, frequency=3.0)
    perturb_x = fbm(noise, xs, ys, octaves=4, frequency=3.0, persistence=0.5) * 0.05
    perturb_y = fbm(SimplexNoise(seed + 200), xs, ys,
                     octaves=4, frequency=3.0, persistence=0.5) * 0.05

    px = x_coords + perturb_x
    py = y_coords + perturb_y

    # Assign each pixel to nearest plate center
    plate_map = np.zeros((res, res), dtype=np.int32)
    min_dist = np.full((res, res), np.inf)

    for plate in plates:
        cx, cy = plate.center
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        mask = dist < min_dist
        plate_map[mask] = plate.plate_id
        min_dist[mask] = dist[mask]

    # ----- Detect boundaries -----
    print("  [tectonic] Detecting plate boundaries...")
    boundary_map = np.zeros((res, res), dtype=np.float64)

    # Detect edges using gradient of plate_map
    # A boundary exists where neighboring cells have different plate IDs
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(plate_map, dy, axis=0), dx, axis=1)
        boundary_map += (plate_map != shifted).astype(np.float64)

    boundary_map = clamp(boundary_map, 0.0, 1.0)

    # Blur boundaries for smooth transitions
    boundary_map = ndimage.gaussian_filter(boundary_map, sigma=res / 200)
    boundary_map = normalize(boundary_map)

    print("  [tectonic] Done.")
    return plate_map, boundary_map, plates


def compute_collision_map(plate_map: np.ndarray,
                          plates: List[TectonicPlate],
                          config: WorldConfig) -> np.ndarray:
    """
    Compute collision intensity at plate boundaries.

    Convergent boundaries → positive (mountains)
    Divergent boundaries → negative (rifts)
    Transform boundaries → near zero (faults)
    """
    res = config.resolution
    tc = config.tectonic

    print("  [tectonic] Computing collision forces...")

    collision = np.zeros((res, res), dtype=np.float64)
    y_coords, x_coords = np.mgrid[0:res, 0:res] / res

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted_map = np.roll(np.roll(plate_map, dy, axis=0), dx, axis=1)
        boundary = plate_map != shifted_map

        if not np.any(boundary):
            continue

        # For each boundary pixel, compute relative plate motion
        by, bx = np.where(boundary)
        for i in range(0, len(by), max(1, len(by) // 1000)):  # Sample for performance
            y, x = by[i], bx[i]
            p1_id = plate_map[y, x]
            p2_id = shifted_map[y, x]

            if p1_id >= len(plates) or p2_id >= len(plates):
                continue

            p1 = plates[p1_id]
            p2 = plates[p2_id]

            # Direction from p1 to boundary
            bdir_x = dx
            bdir_y = dy
            norm = np.sqrt(bdir_x ** 2 + bdir_y ** 2) + 1e-10

            # Relative velocity projected onto boundary normal
            rel_vx = p1.velocity[0] - p2.velocity[0]
            rel_vy = p1.velocity[1] - p2.velocity[1]
            convergence = (rel_vx * bdir_x + rel_vy * bdir_y) / norm

            # Type-based intensity
            if p1.plate_type == "continental" and p2.plate_type == "continental":
                intensity = convergence * tc.collision_mountain_height
            elif p1.plate_type != p2.plate_type:
                intensity = convergence * 0.6  # Subduction
            else:
                intensity = convergence * 0.3  # Oceanic

            collision[y, x] += intensity

    # Spread collision influence
    collision = ndimage.gaussian_filter(collision, sigma=res / 80)
    return collision


def apply_tectonics(heightmap: np.ndarray,
                    plate_map: np.ndarray,
                    boundary_map: np.ndarray,
                    collision_map: np.ndarray,
                    plates: List[TectonicPlate],
                    config: WorldConfig) -> np.ndarray:
    """
    Apply tectonic effects to the heightmap.

    - Convergent zones → raise terrain (mountains)
    - Divergent zones → lower terrain (rifts, mid-ocean ridges)
    - Subduction zones → create trenches + volcanic arcs
    """
    tc = config.tectonic

    print("  [tectonic] Applying tectonic deformation...")

    result = heightmap.copy()

    # Mountains at convergent boundaries
    convergent = clamp(collision_map, 0.0, 1.0)
    result += convergent * tc.collision_mountain_height * boundary_map

    # Rifts at divergent boundaries
    divergent = clamp(-collision_map, 0.0, 1.0)
    result -= divergent * tc.rift_valley_depth * boundary_map

    # Oceanic trenches at subduction zones
    for y in range(heightmap.shape[0]):
        for x in range(0, heightmap.shape[1], max(1, heightmap.shape[1] // 100)):
            if boundary_map[y, x] > 0.3:
                p_id = plate_map[y, x]
                if p_id < len(plates) and plates[p_id].plate_type == "oceanic":
                    if collision_map[y, x] > 0.2:
                        result[y, x] += tc.subduction_trench_depth * boundary_map[y, x]

    # Add fault roughness
    seed = config.sub_seed("tectonic_faults")
    noise = SimplexNoise(seed)
    xs, ys = make_grid(config.resolution, frequency=8.0)
    fault_noise = fbm(noise, xs, ys, octaves=3, frequency=8.0, persistence=0.6)
    result += fault_noise * tc.fault_roughness * boundary_map * 0.1

    result = normalize(result)
    print("  [tectonic] Done.")
    return result
