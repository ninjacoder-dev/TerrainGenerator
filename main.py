"""
main.py — Point d'entrée CLI du moteur TerrainGenerator.

Exemples d'utilisation :
  python main.py --seed 42 --preset alps --resolution 1024
  python main.py --preset mars --resolution 512
  python main.py --preset desert --output output_desert
"""

import argparse
import sys
from pathlib import Path
import copy
import gc

from config import WorldConfig, get_preset, PRESETS
from world import World
from utils.noise import set_global_offset, reset_global_offset


def parse_args():
    parser = argparse.ArgumentParser(
        description="TerrainGenerator — Moteur de Génération Procédurale de Terrain"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed aléatoire déterministe (defaut: 42)"
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Résolution du terrain (largeur x hauteur, ex: 512, 1024, 2048)"
    )
    parser.add_argument(
        "--preset", type=str, default=None, choices=list(PRESETS.keys()),
        help=f"Preset de monde ({', '.join(PRESETS.keys())})"
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Dossier de sortie des cartes exportées"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Fichier JSON de configuration personnalisé"
    )
    parser.add_argument(
        "--size-side", type=int, default=1,
        help="Générer une grille NxN de tuiles (ex: 10 => 100)"
    )
    parser.add_argument(
        "--export-3d", action="store_true",
        help="Exporter un fichier OBJ 3D pour chaque tuile"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        print(f"Loading config from '{args.config}'...")
        cfg = WorldConfig.load(args.config)
    elif args.preset:
        print(f"Using preset '{args.preset}'...")
        cfg = get_preset(args.preset)
    else:
        cfg = WorldConfig()

    # Overrides from CLI
    cfg.seed = args.seed
    cfg.resolution = args.resolution
    cfg.export.output_dir = args.output
    cfg.export.export_3d = args.export_3d

    size = args.size_side or 1
    if size > 1:
        print(f"Generating grid {size}x{size}...")
        out_base = Path(cfg.export.output_dir)
        out_base.mkdir(parents=True, exist_ok=True)
        total = size * size
        counter = 0
        for ty in range(size):
            for tx in range(size):
                counter += 1
                print(f"\n[grid] Tile {counter}/{total} (x={tx}, y={ty})")
                # Deterministic tile-specific seed
                tile_seed = cfg.sub_seed(f"tile:{tx}:{ty}")
                cfg_tile = copy.deepcopy(cfg)
                cfg_tile.seed = tile_seed

                # Set global coordinate offset for continuity between tiles
                set_global_offset(tx / float(size), ty / float(size))

                # Per-tile output directory
                tile_out = out_base / f"tile_{ty}_{tx}"
                cfg_tile.export.output_dir = str(tile_out)
                tile_out.mkdir(parents=True, exist_ok=True)

                # Generate and export
                w = World(cfg_tile)
                w.generate()

                # Optional 3D export (OBJ)
                if cfg_tile.export.export_3d and "heightmap" in w.maps:
                    try:
                        from exporters.mesh import export_obj
                        export_obj(w.maps["heightmap"], str(tile_out / "terrain.obj"), cfg_tile)
                    except Exception as e:
                        print(f"  [export_3d] Failed to export OBJ: {e}")

                w.export()

                # Unload memory for previous tile
                del w
                gc.collect()

                # Optional memory monitoring (requires psutil)
                try:
                    import psutil
                    rss = psutil.Process().memory_info().rss
                    print(f"  [mem] RSS={rss/1024/1024:.1f}MiB")
                    if rss > 1_000_000_000:
                        print("  [mem] Memory over 1GB — forcing GC")
                        gc.collect()
                except Exception:
                    pass

                # Reset global offsets so subsequent operations not influenced
                reset_global_offset()

        print("Grid generation complete.")
    else:
        # Single generation (original behaviour)
        world = World(cfg)
        world.generate()
        if cfg.export.export_3d and "heightmap" in world.maps:
            try:
                from exporters.mesh import export_obj
                export_obj(world.maps["heightmap"], str(Path(cfg.export.output_dir) / "terrain.obj"), cfg)
            except Exception as e:
                print(f"  [export_3d] Failed to export OBJ: {e}")
        world.export()


if __name__ == "__main__":
    main()
