"""
biome/biome_generator.py — Classification des biomes par diagramme de Whittaker.

Croisement : Altitude + Humidité + Température.

Biomes :
  0: Ocean / Water
  1: Desert
  2: Savanna
  3: Grassland / Prairie
  4: Forest / Woodland
  5: Taiga / Boreal Forest
  6: Tundra
  7: Snow / Ice
  8: Jungle / Tropical Rainforest
"""

import numpy as np
from enum import IntEnum
from typing import Dict, Tuple

from config import WorldConfig


class BiomeType(IntEnum):
    OCEAN = 0
    DESERT = 1
    SAVANNA = 2
    GRASSLAND = 3
    FOREST = 4
    TAIGA = 5
    TUNDRA = 6
    SNOW = 7
    JUNGLE = 8


BIOME_COLORS: Dict[int, Tuple[int, int, int]] = {
    BiomeType.OCEAN: (28, 107, 160),
    BiomeType.DESERT: (230, 200, 130),
    BiomeType.SAVANNA: (180, 190, 90),
    BiomeType.GRASSLAND: (100, 180, 70),
    BiomeType.FOREST: (40, 130, 50),
    BiomeType.TAIGA: (30, 90, 70),
    BiomeType.TUNDRA: (140, 160, 150),
    BiomeType.SNOW: (240, 245, 255),
    BiomeType.JUNGLE: (10, 100, 40),
}


def classify_biomes(heightmap: np.ndarray,
                    humidity: np.ndarray,
                    temperature: np.ndarray,
                    sea_level: float) -> np.ndarray:
    """
    Classify each cell into a biome index based on Whittaker diagram.
    """
    h, w = heightmap.shape
    biomes = np.full((h, w), BiomeType.GRASSLAND, dtype=np.int32)

    # Ocean mask
    ocean_mask = heightmap <= sea_level
    biomes[ocean_mask] = BiomeType.OCEAN

    land_mask = ~ocean_mask

    # High altitude / cold snow
    snow_mask = land_mask & ((heightmap > 0.8) | (temperature < 0.15))
    biomes[snow_mask] = BiomeType.SNOW

    # Tundra (cold & dry/medium)
    tundra_mask = land_mask & ~snow_mask & (temperature < 0.3)
    biomes[tundra_mask] = BiomeType.TUNDRA

    # Taiga (cool & medium moisture)
    taiga_mask = land_mask & ~snow_mask & ~tundra_mask & (temperature < 0.45) & (humidity >= 0.3)
    biomes[taiga_mask] = BiomeType.TAIGA

    # Desert (hot & very dry)
    desert_mask = land_mask & ~snow_mask & ~tundra_mask & (temperature >= 0.45) & (humidity < 0.25)
    biomes[desert_mask] = BiomeType.DESERT

    # Savanna (warm & dry-medium)
    savanna_mask = land_mask & ~snow_mask & ~tundra_mask & ~desert_mask & (temperature >= 0.6) & (humidity < 0.5)
    biomes[savanna_mask] = BiomeType.SAVANNA

    # Jungle (hot & very wet)
    jungle_mask = land_mask & ~snow_mask & ~tundra_mask & ~desert_mask & (temperature >= 0.65) & (humidity >= 0.75)
    biomes[jungle_mask] = BiomeType.JUNGLE

    # Forest (temperate & wet)
    forest_mask = land_mask & ~snow_mask & ~tundra_mask & ~desert_mask & ~savanna_mask & ~jungle_mask & (humidity >= 0.5)
    biomes[forest_mask] = BiomeType.FOREST

    return biomes
