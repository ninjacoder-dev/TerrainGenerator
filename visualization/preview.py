"""
visualization/preview.py — Rendu et prévisualisation 3D temps réel / 2D.

Propose un mode de prévisualisation 2D rapide matplotlib
et la structure de rendu 3D ModernGL.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def show_2d_preview(world_dict: dict, save_path: str = None) -> None:
    """
    Display a 2x4 grid preview of generated maps using matplotlib.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("TerrainGenerator — World Preview", fontsize=16)

    maps_to_show = [
        ("Heightmap", world_dict.get("heightmap"), "terrain"),
        ("Continent Mask", world_dict.get("continent_mask"), "Blues_r"),
        ("Slope", world_dict.get("slope_map"), "magma"),
        ("Curvature", world_dict.get("curvature_map"), "coolwarm"),
        ("Humidity", world_dict.get("humidity_map"), "YlGnBu"),
        ("Temperature", world_dict.get("temperature_map"), "inferno"),
        ("Hydraulic Erosion", world_dict.get("sediment_map"), "copper"),
        ("Water / Rivers", world_dict.get("water_mask"), "Blues"),
    ]

    for ax, (title, data, cmap) in zip(axes.flat, maps_to_show):
        if data is not None:
            im = ax.imshow(data, cmap=cmap, origin="upper")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.7)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  [preview] Saved 2D Preview grid: {save_path}")
    else:
        plt.show()
