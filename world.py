"""
world.py — Gestionnaire de la génération complète du monde.

Orchestre toutes les étapes du pipeline procédural :
  1. Continents
  2. Plaques tectoniques
  3. Relief de base (bruit)
  4. Domaine warping & montagnes, volcans, vallées, canyons, plaines, dunes, glaciers
  5. Érosions (thermique, hydraulique, éolienne, glaciaire)
  6. Masques (pente, courbure, altitude, humidité, température, eau)
  7. Biomes & végétation/rochers/neige
  8. Hydrologie (rivières, lacs, océan)
  9. Exports
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from config import WorldConfig
from generators.continent import generate_continent_mask
from generators.tectonic import generate_tectonic_plates, compute_collision_map, apply_tectonics
from generators.base_noise import generate_base_noise
from generators.mountains import generate_mountains
from generators.volcano import generate_volcanos
from generators.valleys import generate_valleys
from generators.canyons import generate_canyons
from generators.plains import generate_plains
from generators.dunes import generate_dunes
from generators.glaciers import generate_glaciers
from erosion.thermal import simulate_thermal_erosion
from erosion.hydraulic import simulate_hydraulic_erosion
from erosion.wind import simulate_wind_erosion
from erosion.glacier import simulate_glacial_erosion
from erosion.sediment import generate_sediment_map
from erosion.weathering import simulate_weathering
from masks.slope import generate_slope_mask
from masks.altitude import generate_altitude_mask
from masks.curvature import generate_curvature_mask
from masks.humidity import generate_humidity_mask
from masks.temperature import generate_temperature_mask
from masks.water import generate_water_mask
from masks.biome import generate_biome_mask
from biome.rocks import generate_rock_mask
from biome.forests import generate_forest_mask
from biome.snow import generate_snow_mask
from biome.grass import generate_grass_mask
from biome.desert import generate_desert_mask
from generators.rivers import generate_rivers
from generators.lakes import generate_lakes
from generators.ocean import generate_ocean
from exporters.png16 import export_png16
from exporters.normal import export_normal_map
from exporters.raw import export_raw
from exporters.tif import export_tif
from exporters.splatmap import export_splatmap
from visualization.preview import show_2d_preview


class World:
    """
    Conteneur et orchestrateur de génération procédurale.
    """

    def __init__(self, config: WorldConfig):
        self.config = config
        self.maps: Dict[str, np.ndarray] = {}

    def generate(self) -> Dict[str, np.ndarray]:
        """Run the full procedural pipeline."""
        start_time = time.time()
        print(f"\n=======================================================")
        print(f" TerrainGenerator — Generating World (Seed: {self.config.seed})")
        print(f" Resolution: {self.config.resolution}x{self.config.resolution}")
        print(f"=======================================================\n")

        cfg = self.config

        # 1. Continent Mask
        if cfg.is_stage_enabled("continent"):
            self.maps["continent_mask"] = generate_continent_mask(cfg)
        else:
            self.maps["continent_mask"] = np.ones((cfg.resolution, cfg.resolution))

        # 2. Base Noise
        if cfg.is_stage_enabled("base_noise"):
            heightmap = generate_base_noise(cfg, self.maps["continent_mask"])
        else:
            heightmap = np.zeros((cfg.resolution, cfg.resolution))

        # 3. Tectonic Plates
        if cfg.is_stage_enabled("tectonic"):
            plate_map, boundary_map, plates = generate_tectonic_plates(cfg)
            collision_map = compute_collision_map(plate_map, plates, cfg)
            heightmap = apply_tectonics(heightmap, plate_map, boundary_map, collision_map, plates, cfg)
            self.maps["boundary_map"] = boundary_map
        else:
            boundary_map = None

        # 4. Feature Generators
        if cfg.is_stage_enabled("mountains"):
            heightmap = generate_mountains(cfg, heightmap, boundary_map)

        if cfg.is_stage_enabled("volcano"):
            heightmap = generate_volcanos(cfg, heightmap, boundary_map)

        if cfg.is_stage_enabled("valleys"):
            heightmap = generate_valleys(cfg, heightmap)

        if cfg.is_stage_enabled("canyons"):
            heightmap = generate_canyons(cfg, heightmap)

        if cfg.is_stage_enabled("plains"):
            heightmap = generate_plains(cfg, heightmap)

        if cfg.is_stage_enabled("dunes"):
            heightmap = generate_dunes(cfg, heightmap)

        glacier_ice = None
        if cfg.is_stage_enabled("glaciers"):
            heightmap, glacier_ice = generate_glaciers(cfg, heightmap)

        # 5. Erosion Pipeline
        if cfg.is_stage_enabled("thermal_erosion"):
            heightmap = simulate_thermal_erosion(cfg, heightmap)

        sediment_map = np.zeros_like(heightmap)
        if cfg.is_stage_enabled("hydraulic_erosion"):
            heightmap, sediment_map = simulate_hydraulic_erosion(cfg, heightmap)

        if cfg.is_stage_enabled("wind_erosion"):
            heightmap = simulate_wind_erosion(cfg, heightmap)

        if cfg.is_stage_enabled("glacial_erosion") and glacier_ice is not None:
            heightmap = simulate_glacial_erosion(cfg, heightmap, glacier_ice)

        if cfg.is_stage_enabled("weathering"):
            heightmap = simulate_weathering(cfg, heightmap)

        self.maps["heightmap"] = heightmap
        self.maps["sediment_map"] = generate_sediment_map(heightmap, sediment_map)

        # 6. Hydrology & Oceans
        if cfg.is_stage_enabled("ocean"):
            heightmap, ocean_mask = generate_ocean(cfg, heightmap)
            self.maps["ocean_mask"] = ocean_mask

        if cfg.is_stage_enabled("rivers"):
            heightmap, river_mask = generate_rivers(cfg, heightmap)
            self.maps["river_mask"] = river_mask
        else:
            river_mask = None

        if cfg.is_stage_enabled("lakes"):
            heightmap, lake_mask = generate_lakes(cfg, heightmap, river_mask)
            self.maps["lake_mask"] = lake_mask
        else:
            lake_mask = None

        self.maps["heightmap"] = heightmap

        # 7. Masks & Biomes
        if cfg.is_stage_enabled("masks"):
            self.maps["slope_map"] = generate_slope_mask(heightmap)
            self.maps["altitude_map"] = generate_altitude_mask(heightmap)
            self.maps["curvature_map"] = generate_curvature_mask(heightmap)
            self.maps["water_mask"] = generate_water_mask(heightmap, cfg.sea_level, river_mask, lake_mask)
            self.maps["humidity_map"] = generate_humidity_mask(cfg, heightmap, self.maps["water_mask"])
            self.maps["temperature_map"] = generate_temperature_mask(cfg, heightmap)

        if cfg.is_stage_enabled("biomes"):
            self.maps["biome_map"] = generate_biome_mask(
                heightmap, self.maps["humidity_map"], self.maps["temperature_map"], cfg.sea_level
            )
            self.maps["rock_mask"] = generate_rock_mask(heightmap, self.maps["slope_map"], cfg.sea_level, cfg.seed)
            self.maps["forest_mask"] = generate_forest_mask(self.maps["biome_map"], self.maps["slope_map"], self.maps["humidity_map"])
            self.maps["snow_mask"] = generate_snow_mask(heightmap, self.maps["temperature_map"], self.maps["curvature_map"])
            self.maps["grass_mask"] = generate_grass_mask(self.maps["biome_map"], self.maps["slope_map"])
            self.maps["desert_mask"] = generate_desert_mask(self.maps["biome_map"])

        elapsed = time.time() - start_time
        print(f"\n=======================================================")
        print(f" Generation completed in {elapsed:.2f} seconds!")
        print(f"=======================================================\n")

        return self.maps

    def export(self, output_dir: Optional[str] = None) -> None:
        """Export generated maps to disk."""
        out = output_dir or self.config.export.output_dir
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"  [export] Saving all maps to directory: '{out}'...")

        if "heightmap" in self.maps:
            export_png16(self.maps["heightmap"], str(out_path / "heightmap.png"))
            export_normal_map(self.maps["heightmap"], str(out_path / "normal_map.png"))
            export_raw(self.maps["heightmap"], str(out_path / "heightmap.raw"))
            export_tif(self.maps["heightmap"], str(out_path / "heightmap.tif"))

        for name in ["continent_mask", "slope_map", "curvature_map", "humidity_map",
                     "temperature_map", "water_mask", "sediment_map", "rock_mask",
                     "snow_mask", "forest_mask", "grass_mask"]:
            if name in self.maps:
                export_png16(self.maps[name], str(out_path / f"{name}.png"))

        if all(k in self.maps for k in ["rock_mask", "grass_mask", "desert_mask", "snow_mask"]):
            export_splatmap(
                self.maps["rock_mask"], self.maps["grass_mask"],
                self.maps["desert_mask"], self.maps["snow_mask"],
                str(out_path / "splatmap.png")
            )

        # Save config
        self.config.save(str(out_path / "world_config.json"))

        # Save preview grid
        show_2d_preview(self.maps, save_path=str(out_path / "preview_grid.png"))

        print(f"  [export] Export finished successfully.")
