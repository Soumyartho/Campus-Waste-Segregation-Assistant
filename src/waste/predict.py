"""Inference (PLAN.md Section 7.4): top-k probabilities and the three confidence states.

Usage: python -m waste.predict path/to/image.jpg [more images ...]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from waste.data.transforms import eval_transform
from waste.utils import load_yaml


class State(str, Enum):
    """Confidence state of a prediction."""

    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    UNRECOGNISED = "unrecognised"


@dataclass
class Prediction:
    """Result of classifying one image."""

    state: State
    label: str
    confidence: float
    top_k: list[tuple[str, float]]
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "label": self.label, "confidence": self.confidence,
                "top_k": self.top_k}


def classify_state(top_prob: float, tau: float, reject: float) -> State:
    """Confident at or above tau, unrecognised below reject, uncertain in between."""
    if top_prob >= tau:
        return State.CONFIDENT
    if top_prob >= reject:
        return State.UNCERTAIN
    return State.UNRECOGNISED


def prediction_from_probs(
    probs: np.ndarray, class_names: list[str], tau: float, reject: float, top_k: int = 3
) -> Prediction:
    """Build a Prediction from one probability vector."""
    order = np.argsort(-probs)[:top_k]
    top = [(class_names[i], float(probs[i])) for i in order]
    return Prediction(
        state=classify_state(top[0][1], tau, reject),
        label=top[0][0],
        confidence=top[0][1],
        top_k=top,
        probabilities={c: float(p) for c, p in zip(class_names, probs, strict=True)},
    )


class Predictor:
    """Loads a checkpoint once and classifies PIL images on CPU (or a chosen device)."""

    def __init__(
        self,
        model: nn.Module,
        meta: dict[str, Any],
        tau: float,
        reject: float,
        top_k: int = 3,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        if not 0 <= reject <= tau <= 1:
            raise ValueError(
                f"Thresholds must satisfy 0 <= reject <= tau <= 1 (got {reject}, {tau})"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.meta = meta
        self.class_names: list[str] = meta["class_names"]
        self.model_name: str = meta.get("config", {}).get("run_name", meta["model_name"])
        self.tau, self.reject, self.top_k, self.temperature = tau, reject, top_k, temperature
        self.transform = eval_transform()

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/inference.yaml") -> Predictor:
        """Build a predictor from inference.yaml; raises FileNotFoundError if no checkpoint."""
        from waste.checkpoint import checkpoint_path, load_checkpoint

        cfg = load_yaml(config_path)
        m, th = cfg["model"], cfg["thresholds"]
        model, meta = load_checkpoint(checkpoint_path(m["models_dir"], m["run_name"]), m["device"])
        return cls(model, meta, th["tau"], th["reject"], th["top_k"],
                   cfg.get("temperature", 1.0), m["device"])

    @torch.inference_mode()
    def probabilities(self, images: list[Image.Image]) -> np.ndarray:
        """Softmax probabilities (temperature-scaled) for a list of images."""
        x = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        logits = self.model(x).float() / self.temperature
        return torch.softmax(logits, 1).cpu().numpy()

    def predict(self, image: Image.Image) -> Prediction:
        """Classify one image."""
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[Image.Image], batch_size: int = 32) -> list[Prediction]:
        """Classify many images in mini-batches."""
        out: list[Prediction] = []
        for i in range(0, len(images), batch_size):
            for p in self.probabilities(images[i : i + batch_size]):
                out.append(prediction_from_probs(p, self.class_names, self.tau, self.reject,
                                                 self.top_k))
        return out


def main() -> None:
    from waste.data.dataset import load_image
    from waste.disposal import DisposalEngine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--config", default="configs/inference.yaml")
    args = parser.parse_args()

    predictor = Predictor.from_config(args.config)
    engine = DisposalEngine.from_config(args.config, predictor.class_names)
    for path in args.images:
        pred = predictor.predict(load_image(path))
        rec = engine.recommend(pred)
        tops = ", ".join(f"{c} {p:.0%}" for c, p in pred.top_k)
        print(f"\n{path}\n  [{pred.state.value}] {tops}\n  {rec.title}")
        if rec.rule:
            print(f"  {rec.rule}")
        for line in (rec.primary.steps if rec.primary else rec.advice):
            print(f"   - {line}")


if __name__ == "__main__":
    main()
