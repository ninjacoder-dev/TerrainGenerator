"""
generators/volcano.py — Générateur de volcans.

Types :
  - Bouclier (shield) — large, pente douce
  - Stratovolcan — cône raide, sommet pointu
  - Caldeira — grand cratère effondré
  - Cône volcanique (cinder cone) — petit, très raide

Chaque volcan possède :
  - hauteur, rayon, taille du cratère, activité, coulées de lave
"""

import numpy as np
from typing import List, Tuple, Optional

from config import WorldConfig
from utils.noise import SimplexNoise, make_grid, fbm
from utils.math_utils import normalize, smoothstep, clamp
from utils.random import SeededRandom, poisson_disk_sampling
from utils.filters import gaussian_blur


class Volcano:
    """Représentation d'un volcan individuel."""

    def __init__(self, x: int, y: int, volcano_type: str,
                 height: float, radius: float,
                 crater_ratio: float, activity: float):
        self.x = x
        self.y = y
        self.volcano_type = volcano_type
        self.height = height
        self.radius = radius
        self.crater_ratio = crater_ratio
        self.activity = activity


def generate_volcanos(config: WorldConfig,
                      heightmap: np.ndarray,
                      boundary_map: Optional[np.ndarray] = None
                      ) -> np.ndarray:
    """
    Place and generate volcanos on the heightmap.
    """
    res = config.resolution
    vc = config.volcano
    seed = config.sub_seed("volcano")
    rng = SeededRandom(seed)

    print(f"  [volcano] Generating up to {vc.max_count} volcanos...")

    result = heightmap.copy()

    # ----- Determine volcano locations -----
    # Prefer tectonic boundaries and existing high terrain
    volcanos = _place_volcanos(config, heightmap, boundary_map, rng)

    # ----- Generate each volcano -----
    for i, volcano in enumerate(volcanos):
        print(f"  [volcano] Building {volcano.volcano_type} at ({volcano.x}, {volcano.y})...")
        cone = _build_volcano(volcano, res, seed + i * 100)
        result = np.maximum(result, result + cone * 0.5)

    result = normalize(result)
    print(f"  [volcano] Done. {len(volcanos)} volcanos placed.")
    return result


def _place_volcanos(config: WorldConfig,
                    heightmap: np.ndarray,
                    boundary_map: Optional[np.ndarray],
                    rng: SeededRandom) -> List[Volcano]:
    """Determine volcano locations and properties."""
    res = config.resolution
    vc = config.volcano
    volcanos = []

    # Use Poisson disk sampling for spacing
    min_dist = res / (vc.max_count + 1) * 0.8
    points = poisson_disk_sampling(
        float(res), float(res), min_dist,
        seed=rng.seed, max_attempts=20
    )

    # Score each point — prefer boundaries
    scored_points = []
    for px, py in points:
        ix, iy = int(clamp(np.array([px]), 0, res - 1)[0]), int(clamp(np.array([py]), 0, res - 1)[0])
        score = 0.3  # Base score
        if boundary_map is not None:
            score += boundary_map[iy, ix] * 0.7
        score += heightmap[iy, ix] * 0.2
        score += rng.uniform(0.0, 0.3)
        scored_points.append((score, ix, iy))

    scored_points.sort(reverse=True)

    for score, x, y in scored_points[:vc.max_count]:
        if score < 0.3:
            continue

        # Choose type
        v_type = rng.choice(vc.types)
        height = rng.uniform(0.3, 1.0) * vc.max_height
        radius = rng.uniform(res * 0.03, res * 0.08)

        # Type-specific adjustments
        if v_type == "shield":
            radius *= 2.0
            height *= 0.6
            crater = 0.05
        elif v_type == "stratovolcano":
            height *= 1.0
            crater = vc.crater_ratio * 0.8
        elif v_type == "caldera":
            radius *= 1.5
            height *= 0.7
            crater = vc.crater_ratio * 2.0
        else:  # cinder_cone
            radius *= 0.5
            height *= 0.4
            crater = vc.crater_ratio

        activity = rng.uniform(0.0, 1.0)

        volcanos.append(Volcano(
            x=x, y=y, volcano_type=v_type,
            height=height, radius=radius,
            crater_ratio=min(crater, 0.4),
            activity=activity
        ))

    return volcanos


def _build_volcano(volcano: Volcano, resolution: int,
                   seed: int) -> np.ndarray:
    """Build a single volcano cone on the heightmap."""
    result = np.zeros((resolution, resolution), dtype=np.float64)

    y_coords, x_coords = np.mgrid[0:resolution, 0:resolution]
    dist = np.sqrt(
        (x_coords - volcano.x) ** 2 +
        (y_coords - volcano.y) ** 2
    )

    # Normalize distance by radius
    r = dist / (volcano.radius + 1e-10)

    # ----- Cone shape -----
    if volcano.volcano_type == "shield":
        # Gentle parabolic slope
        cone = np.maximum(0, 1.0 - r ** 2) * volcano.height
    elif volcano.volcano_type == "stratovolcano":
        # Steep exponential slope
        cone = np.maximum(0, np.exp(-r * 3.0) - 0.05) * volcano.height
    elif volcano.volcano_type == "caldera":
        # Ring shape with collapsed center
        cone = np.maximum(0, 1.0 - r ** 1.5) * volcano.height
        # Collapse the center
        inner_r = r / volcano.crater_ratio
        collapse = smoothstep(0.0, 1.0, inner_r) * 0.6
        cone = cone * collapse + cone * (1.0 - collapse) * 0.4
    else:  # cinder_cone
        # Sharp conical
        cone = np.maximum(0, 1.0 - r) * volcano.height

    # ----- Crater -----
    if volcano.volcano_type != "caldera":
        crater_r = r / volcano.crater_ratio
        crater_mask = smoothstep(0.8, 1.0, 1.0 - crater_r)
        crater_depth = crater_mask * volcano.height * 0.3
        cone = cone - crater_depth
        cone = np.maximum(0, cone)

    # ----- Add noise for natural irregularity -----
    noise = SimplexNoise(seed)
    xs, ys = make_grid(resolution, frequency=8.0)
    detail = fbm(noise, xs, ys, octaves=4, frequency=8.0, persistence=0.5)
    detail = normalize(detail) * 0.1

    # Only apply noise on the cone surface
    cone_mask = (cone > 0.01).astype(np.float64)
    cone += detail * cone_mask * volcano.height * 0.15

    # ----- Lava flows (for active volcanos) -----
    if volcano.activity > 0.5:
        lava = _generate_lava_flows(volcano, resolution, seed + 50)
        cone += lava * 0.1

    return np.maximum(0, cone)


def _generate_lava_flows(volcano: Volcano, resolution: int,
                         seed: int) -> np.ndarray:
    """Generate lava flow patterns radiating from volcano."""
    noise = SimplexNoise(seed)
    y_coords, x_coords = np.mgrid[0:resolution, 0:resolution]

    # Angle from volcano center
    angle = np.arctan2(y_coords - volcano.y, x_coords - volcano.x)
    dist = np.sqrt(
        (x_coords - volcano.x) ** 2 +
        (y_coords - volcano.y) ** 2
    )

    # Create radial channels using noise
    r = dist / (volcano.radius * 2 + 1e-10)
    flow_pattern = np.sin(angle * 5 + seed * 0.1) * 0.5 + 0.5

    xs, ys = make_grid(resolution, frequency=6.0)
    flow_noise = fbm(noise, xs, ys, octaves=3, frequency=6.0, persistence=0.5)
    flow_noise = normalize(flow_noise)

    lava = flow_pattern * flow_noise * np.maximum(0, 1.0 - r)
    lava = clamp(lava, 0.0, 1.0)

    return lava * volcano.activity
