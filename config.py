"""
config.py — Configuration globale du moteur TerrainGenerator.

Contient la classe WorldConfig avec tous les paramètres du monde,
le système de presets, et la gestion des seeds déterministes.
"""

import hashlib
import json
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """Configuration pour une couche de bruit."""
    noise_type: str = "opensimplex"  # opensimplex, ridged, billow, worley, fbm
    octaves: int = 8
    frequency: float = 1.0
    amplitude: float = 1.0
    persistence: float = 0.5
    lacunarity: float = 2.0
    offset: float = 1.0  # For ridged noise


@dataclass
class WarpConfig:
    """Configuration du Domain Warping."""
    enabled: bool = True
    strength: float = 80.0
    octaves: int = 4
    frequency: float = 0.5
    iterations: int = 2  # Recursive warping iterations


@dataclass
class ContinentConfig:
    """Configuration de la génération des continents."""
    enabled: bool = True
    sea_level: float = 0.4
    continent_frequency: float = 0.8
    island_frequency: float = 3.0
    island_density: float = 0.3
    coast_smoothing: float = 5.0
    worley_weight: float = 0.3


@dataclass
class TectonicConfig:
    """Configuration des plaques tectoniques."""
    enabled: bool = True
    num_plates: int = 12
    collision_mountain_height: float = 0.8
    subduction_trench_depth: float = -0.3
    rift_valley_depth: float = -0.15
    fault_roughness: float = 0.5


@dataclass
class MountainConfig:
    """Configuration du générateur de montagnes."""
    enabled: bool = True
    types: List[str] = field(default_factory=lambda: [
        "alpine", "himalayan", "rocky", "volcanic", "plateau", "hills"
    ])
    max_height: float = 1.0
    ridge_sharpness: float = 2.0
    slope_steepness: float = 1.5


@dataclass
class VolcanoConfig:
    """Configuration du générateur de volcans."""
    enabled: bool = True
    max_count: int = 5
    types: List[str] = field(default_factory=lambda: [
        "shield", "stratovolcano", "caldera", "cinder_cone"
    ])
    max_height: float = 0.9
    crater_ratio: float = 0.15
    lava_flow_length: float = 0.3


@dataclass
class ValleyConfig:
    """Configuration des vallées."""
    enabled: bool = True
    glacial_width: float = 0.3
    fluvial_depth: float = 0.2
    tectonic_scale: float = 0.5


@dataclass
class CanyonConfig:
    """Configuration des canyons."""
    enabled: bool = True
    depth: float = 0.4
    sinuosity: float = 0.6
    wall_steepness: float = 0.9


@dataclass
class DuneConfig:
    """Configuration des dunes."""
    enabled: bool = True
    wind_direction: float = 45.0  # degrees
    wind_strength: float = 0.5
    dune_height: float = 0.1
    dune_spacing: float = 0.05


@dataclass
class GlacierConfig:
    """Configuration des glaciers."""
    enabled: bool = True
    altitude_threshold: float = 0.75
    flow_rate: float = 0.01
    moraine_height: float = 0.05


@dataclass
class ThermalErosionConfig:
    """Configuration de l'érosion thermique."""
    enabled: bool = True
    iterations: int = 50
    talus_angle: float = 0.6  # angle de repos (en tant que pente normalisée)
    erosion_rate: float = 0.5


@dataclass
class HydraulicErosionConfig:
    """Configuration de l'érosion hydraulique."""
    enabled: bool = True
    num_droplets: int = 500_000
    erosion_rate: float = 0.3
    deposition_rate: float = 0.3
    evaporation_rate: float = 0.01
    gravity: float = 10.0
    min_slope: float = 0.01
    inertia: float = 0.3
    capacity_factor: float = 8.0
    max_lifetime: int = 64
    erosion_radius: int = 3


@dataclass
class WindErosionConfig:
    """Configuration de l'érosion éolienne."""
    enabled: bool = True
    iterations: int = 100
    wind_direction: float = 45.0
    wind_strength: float = 0.3
    abrasion_rate: float = 0.1
    transport_rate: float = 0.3


@dataclass
class GlacialErosionConfig:
    """Configuration de l'érosion glaciaire."""
    enabled: bool = True
    iterations: int = 30
    abrasion_rate: float = 0.2
    plucking_rate: float = 0.1


@dataclass
class HumidityConfig:
    """Configuration de la simulation d'humidité."""
    evaporation_rate: float = 0.3
    wind_direction: float = 225.0
    rainfall_intensity: float = 0.5
    rain_shadow_strength: float = 0.7


@dataclass
class TemperatureConfig:
    """Configuration de la température."""
    base_temperature: float = 30.0  # °C à latitude 0 et altitude 0
    altitude_lapse_rate: float = 6.5  # °C per 1000m
    latitude_gradient: float = 0.7
    water_moderation: float = 0.3


@dataclass
class RiverConfig:
    """Configuration des rivières."""
    enabled: bool = True
    flow_threshold: float = 0.002
    max_rivers: int = 50
    meander_strength: float = 0.3
    delta_spread: float = 0.2


@dataclass
class LakeConfig:
    """Configuration des lacs."""
    enabled: bool = True
    min_area: int = 100  # pixels minimum
    fill_depth: float = 0.02


@dataclass
class ExportConfig:
    """Configuration de l'export."""
    output_dir: str = "output"
    height_map: bool = True
    normal_map: bool = True
    slope_map: bool = True
    curvature_map: bool = True
    flow_map: bool = True
    water_map: bool = True
    biome_map: bool = True
    moisture_map: bool = True
    temperature_map: bool = True
    ao_map: bool = True
    sediment_map: bool = True
    rock_mask: bool = True
    snow_mask: bool = True
    forest_mask: bool = True
    grass_mask: bool = True
    splat_map: bool = True
    format: str = "png16"  # png16, exr, raw, tif
    export_3d: bool = False  # Export a 3D OBJ file


# ---------------------------------------------------------------------------
# Main WorldConfig
# ---------------------------------------------------------------------------

@dataclass
class WorldConfig:
    """Configuration complète du monde."""

    # Core
    seed: int = 42
    resolution: int = 1024  # width = height
    world_size: float = 10000.0  # meters
    sea_level: float = 0.35

    # Pipeline control — which stages to run
    stages: List[str] = field(default_factory=lambda: [
        "continent", "tectonic", "base_noise", "domain_warp",
        "mountains", "volcano", "valleys", "canyons", "plains",
        "dunes", "glaciers",
        "thermal_erosion", "hydraulic_erosion", "wind_erosion",
        "glacial_erosion", "weathering",
        "masks", "biomes", "rivers", "lakes", "ocean",
        "rocks", "forests", "snow", "grass",
        "export"
    ])

    # Sub-configs
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    continent: ContinentConfig = field(default_factory=ContinentConfig)
    tectonic: TectonicConfig = field(default_factory=TectonicConfig)
    mountains: MountainConfig = field(default_factory=MountainConfig)
    volcano: VolcanoConfig = field(default_factory=VolcanoConfig)
    valley: ValleyConfig = field(default_factory=ValleyConfig)
    canyon: CanyonConfig = field(default_factory=CanyonConfig)
    dune: DuneConfig = field(default_factory=DuneConfig)
    glacier: GlacierConfig = field(default_factory=GlacierConfig)
    thermal_erosion: ThermalErosionConfig = field(default_factory=ThermalErosionConfig)
    hydraulic_erosion: HydraulicErosionConfig = field(default_factory=HydraulicErosionConfig)
    wind_erosion: WindErosionConfig = field(default_factory=WindErosionConfig)
    glacial_erosion: GlacialErosionConfig = field(default_factory=GlacialErosionConfig)
    humidity: HumidityConfig = field(default_factory=HumidityConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    river: RiverConfig = field(default_factory=RiverConfig)
    lake: LakeConfig = field(default_factory=LakeConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # -----------------------------------------------------------------------
    # Seed management
    # -----------------------------------------------------------------------

    def sub_seed(self, stage_name: str) -> int:
        """
        Derive a deterministic sub-seed for a specific generation stage.
        Same (seed, stage_name) → same sub-seed, always.
        """
        h = hashlib.sha256(f"{self.seed}:{stage_name}".encode()).hexdigest()
        return int(h[:8], 16)

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "WorldConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "WorldConfig":
        """Recursively build config from dict."""
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
                    sub_cls = type(attr)
                    setattr(cfg, key, sub_cls(**value))
                else:
                    setattr(cfg, key, value)
        return cfg

    def is_stage_enabled(self, stage: str) -> bool:
        return stage in self.stages

    def __repr__(self) -> str:
        return (
            f"WorldConfig(seed={self.seed}, resolution={self.resolution}, "
            f"stages={len(self.stages)} active)"
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {}


def _register_preset(name: str, overrides: Dict[str, Any]):
    PRESETS[name] = overrides


def get_preset(name: str) -> WorldConfig:
    """Create a WorldConfig from a named preset."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    cfg = WorldConfig()
    overrides = PRESETS[name]
    for key, value in overrides.items():
        if hasattr(cfg, key):
            attr = getattr(cfg, key)
            if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    setattr(attr, sub_key, sub_val)
            else:
                setattr(cfg, key, value)
    return cfg


# --- Built-in presets ---

_register_preset("alps", {
    "sea_level": 0.25,
    "noise": {"octaves": 10, "persistence": 0.55, "lacunarity": 2.1},
    "warp": {"strength": 100.0, "iterations": 3},
    "mountains": {"max_height": 1.0, "ridge_sharpness": 3.0, "slope_steepness": 2.0},
    "thermal_erosion": {"iterations": 80, "talus_angle": 0.5},
    "hydraulic_erosion": {"num_droplets": 1_000_000, "erosion_rate": 0.4},
    "glacier": {"altitude_threshold": 0.7, "flow_rate": 0.02},
})

_register_preset("mars", {
    "sea_level": 0.0,
    "continent": {"enabled": False},
    "noise": {"octaves": 6, "persistence": 0.45, "frequency": 0.8},
    "warp": {"strength": 40.0, "iterations": 1},
    "mountains": {"max_height": 0.95, "ridge_sharpness": 1.5},
    "volcano": {"max_count": 3, "max_height": 1.0, "crater_ratio": 0.25},
    "canyon": {"depth": 0.6, "sinuosity": 0.3, "wall_steepness": 0.95},
    "thermal_erosion": {"iterations": 100},
    "hydraulic_erosion": {"enabled": False},
    "river": {"enabled": False},
    "lake": {"enabled": False},
    "stages": [
        "base_noise", "domain_warp", "mountains", "volcano",
        "canyons", "thermal_erosion", "wind_erosion",
        "masks", "export"
    ],
})

_register_preset("volcanic_islands", {
    "sea_level": 0.5,
    "continent": {"sea_level": 0.55, "island_density": 0.7, "continent_frequency": 1.5},
    "noise": {"octaves": 8, "persistence": 0.5},
    "volcano": {"max_count": 15, "max_height": 0.85},
    "hydraulic_erosion": {"num_droplets": 800_000},
})

_register_preset("desert", {
    "sea_level": 0.1,
    "noise": {"octaves": 6, "persistence": 0.4, "frequency": 0.6},
    "warp": {"strength": 50.0},
    "mountains": {"max_height": 0.5},
    "dune": {"wind_strength": 0.8, "dune_height": 0.15},
    "wind_erosion": {"iterations": 200, "wind_strength": 0.6},
    "hydraulic_erosion": {"num_droplets": 100_000},
    "river": {"max_rivers": 5},
})

_register_preset("snowy_world", {
    "sea_level": 0.3,
    "noise": {"octaves": 9, "persistence": 0.5},
    "mountains": {"max_height": 0.9},
    "glacier": {"altitude_threshold": 0.4, "flow_rate": 0.03},
    "glacial_erosion": {"iterations": 60, "abrasion_rate": 0.3},
    "temperature": {"base_temperature": 5.0},
})

_register_preset("archipelago", {
    "sea_level": 0.55,
    "continent": {"sea_level": 0.6, "island_density": 0.8, "continent_frequency": 2.0,
                   "island_frequency": 5.0},
    "noise": {"octaves": 7, "persistence": 0.5},
    "warp": {"strength": 60.0},
    "mountains": {"max_height": 0.6},
    "hydraulic_erosion": {"num_droplets": 600_000},
})

_register_preset("flat_test", {
    "sea_level": 0.3,
    "resolution": 512,
    "noise": {"octaves": 4, "persistence": 0.4, "amplitude": 0.5},
    "warp": {"enabled": False},
    "mountains": {"enabled": False},
    "volcano": {"enabled": False},
    "canyon": {"enabled": False},
    "stages": ["continent", "base_noise", "thermal_erosion", "masks", "export"],
})
