"""Shared fixtures: synthetic images and a tiny trained-shape checkpoint (no dataset needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from waste.utils import class_names_from_map

CLASS_NAMES = class_names_from_map("configs/class_map.yaml")


def make_image(seed: int, size: int = 64) -> Image.Image:
    """A random but deterministic RGB image with coarse structure (so pHashes differ)."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(coarse).resize((size, size), Image.Resampling.NEAREST)


@pytest.fixture
def class_names() -> list[str]:
    return list(CLASS_NAMES)


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    """Folder-per-class dataset with 12 distinct images per class."""
    root = tmp_path / "images"
    for c_idx, cls in enumerate(CLASS_NAMES):
        (root / cls).mkdir(parents=True)
        for i in range(12):
            make_image(1000 * c_idx + i).save(root / cls / f"{cls}_{i}.png")
    return root
