"""Shared helpers: config loading, path resolution, seeding, devices and run metadata."""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path; relative paths are taken from the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file (relative paths resolve against the project root)."""
    with open(resolve_path(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file (relative paths resolve against the project root)."""
    with open(resolve_path(path), encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> Path:
    """Write JSON with indentation, creating parent folders."""
    out = ensure_dir(resolve_path(path).parent) / Path(path).name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return out


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it as an absolute path."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preferred: str = "auto") -> Any:
    """Return a torch.device: 'auto' picks CUDA when available, otherwise CPU."""
    import torch

    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a repo or before a commit."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def class_names_from_map(class_map_path: str | Path) -> list[str]:
    """Return the ordered final class names from class_map.yaml."""
    return list(load_yaml(class_map_path)["classes"].keys())


def normalise_folder_name(name: str) -> str:
    """Normalise a source folder name so 'Brown_Glass', 'brown-glass' and 'brown glass' match."""
    return name.strip().lower().replace("_", "-").replace(" ", "-")
