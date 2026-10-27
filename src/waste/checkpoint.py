"""Save and load model checkpoints together with the metadata needed to rebuild them."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from waste.models.factory import build_model
from waste.utils import git_commit, resolve_path


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_name: str,
    model_kwargs: dict[str, Any],
    class_names: list[str],
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Write weights, class names, config, git commit and validation metrics to ``path``."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "class_names": class_names,
            "config": config,
            "git_commit": git_commit(),
            "metrics": metrics,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        p,
    )
    return p


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint (no pretrained download) and return (model, metadata)."""
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    ckpt = torch.load(p, map_location=device, weights_only=False)
    kwargs = {**ckpt["model_kwargs"], "pretrained": False}
    model = build_model(ckpt["model_name"], len(ckpt["class_names"]), **kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    meta["path"] = str(p)
    meta["size_mb"] = p.stat().st_size / 1e6
    return model, meta


def checkpoint_path(models_dir: str | Path, run_name: str) -> Path:
    """Standard location of a run's best checkpoint: models/<run_name>/best.pt."""
    return resolve_path(models_dir) / run_name / "best.pt"
