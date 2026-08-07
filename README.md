# TerrainGenerator

Moteur procédural de génération de terrains (Python).

## Vue d'ensemble

Ce projet génère des heightmaps, masques et textures (normal, splatmap, etc.) via un pipeline procédural : continents → bruit de base → tectonique / montagnes / volcans → érosion → masques → biomes → export.

Point d'entrée : `main.py`.

## Prérequis

- Python 3.8+
- Paquets pip recommandés :
  - numpy
  - pillow
  - matplotlib
  - tifffile
  - imageio
  - opensimplex (optionnel, améliore la qualité/performance du bruit)

Installation rapide :

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

> Remarque : `opensimplex` est optionnel ; le module a une implémentation de secours si absent.

## Exécution

Se placer à la racine du projet ( dossier contenant main.py )

Affichage de l'aide :

```cmd
python main.py --help
```

Exemples :

```cmd
# Génération avec preset "alps"
python main.py --preset alps --resolution 1024 --seed 42 --output C:\temp\terrain_alps

# Utiliser un fichier de configuration JSON personnalisé
python main.py --config C:\chemin\ma_config.json --output C:\temp\custom

# Exécution rapide (test)
python main.py --preset flat_test --resolution 256 --output output_quick
```

Options CLI principales (valeurs par défaut reflétées dans `main.py`) :
- `--seed` : seed déterministe (defaut: 42)
- `--resolution` : largeur=hauteur (défaut: 512)
- `--preset` : choisir un preset intégré
- `--output` : dossier de sortie (défaut: `output`)
- `--config` : chemin vers un JSON de configuration personnalisé
- `--export-3d  ` : Exporter un fichier OBJ 3D pour chaque tuile
- `--size-side` : permet de créé plusieurs tuile avec plusieurs génération différente, afin de crée un terrain plus complet, exemple : si on met 2, alors le programme créera 4 tuile, pour créé une grande tuile de coter 2 tuiles.

## Presets intégrés

`alps`, `mars`, `volcanic_islands`, `desert`, `snowy_world`, `archipelago`, `flat_test`

## Fichier de configuration JSON

Le JSON doit correspondre aux attributs de la dataclass `WorldConfig` (voir `world_config_example.json`).



## Sorties

Dans le dossier `--output` (par défaut `output`) :
- `heightmap.png` (16-bit)
- `normal_map.png`
- `heightmap.raw` (raw 16-bit)
- `heightmap.tif`
- `preview_grid.png` (aperçu 2×4)
- `world_config.json` (copie de la configuration utilisée)
- divers masques: `slope_map.png`, `humidity_map.png`, `water_mask.png`, etc.

## Structure du projet

- `main.py` — CLI / point d'entrée
- `config.py` — WorldConfig, presets
- `world.py` — orchestration du pipeline
- `generators/` — continent, montagnes, rivières, etc.
- `erosion/` — modules d'érosion (hydraulique, thermique, vent, glace)
- `masks/` — génération de cartes dérivées (pente, humidité, température, eau)
- `biome/` — masques de végétation, neige, roches
- `exporters/` — exporters PNG/TIFF/RAW/EXR/splatmap
- `visualization/` — preview matplotlib (et stub ModernGL)
- `utils/` — bruit, utilitaires math / random / threading

## Astuces et limitations

- Résolution élevée (1024+) consomme beaucoup de RAM et CPU ; tester d'abord à 256/512.
- Si `imageio`/`tifffile` manquent, certains formats (EXR/TIFF) peuvent échouer ; installer les paquets correspondants.
- Pour utilisation programmatique :

```py
from config import get_preset
from world import World
cfg = get_preset("alps")
cfg.resolution = 512
world = World(cfg)
world.generate()
world.export()
```

## Développement

- Ajouter ou modifier des presets dans `config.py` via `_register_preset`.
- Pour étendre le pipeline, implémenter une nouvelle étape et l'ajouter à `WorldConfig.stages` ou contrôler `stages` dans un fichier de config.

---

Pour toute question supplémentaire ou pour un README en anglais, demander ici.
