"""Disposal recommendation engine (PLAN.md Section 9).

Logic lives here; content (bins, colours, steps, wording) lives in configs/disposal_map.json,
so campus staff can edit instructions without touching the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from waste.predict import Prediction, State
from waste.utils import load_json, load_yaml

REQUIRED_CLASS_KEYS = {
    "display_name": str, "bin": str, "bin_color": str, "bin_label": str,
    "recyclable": bool, "steps": list, "do_not": list, "impact_note": str,
}
REQUIRED_FALLBACK_KEYS = {"title": str, "rule": str, "advice": list}


class DisposalMapError(ValueError):
    """The disposal map is missing classes or has malformed entries."""


@dataclass
class ClassAdvice:
    """Disposal guidance for one class."""

    label: str
    probability: float
    display_name: str
    bin: str
    bin_color: str
    bin_label: str
    recyclable: bool
    steps: list[str]
    do_not: list[str]
    impact_note: str


@dataclass
class Recommendation:
    """What the user should do with the item, given the prediction's confidence state."""

    state: State
    title: str
    primary: ClassAdvice | None = None
    candidates: list[ClassAdvice] = field(default_factory=list)
    rule: str | None = None
    advice: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


def validate_disposal_map(data: dict[str, Any], class_names: list[str]) -> None:
    """Raise DisposalMapError unless every model class and both fallbacks are well-formed."""
    classes = data.get("classes")
    if not isinstance(classes, dict):
        raise DisposalMapError("disposal_map.json must contain a 'classes' object")
    missing = [c for c in class_names if c not in classes]
    if missing:
        raise DisposalMapError(
            f"disposal_map.json has no entry for model classes {missing}. "
            "Add an entry for each class before starting the app."
        )
    problems = []
    for name in class_names:
        for key, typ in REQUIRED_CLASS_KEYS.items():
            if not isinstance(classes[name].get(key), typ):
                problems.append(f"classes.{name}.{key} must be {typ.__name__}")
    fallbacks = data.get("fallbacks", {})
    for state in (State.UNCERTAIN.value, State.UNRECOGNISED.value):
        for key, typ in REQUIRED_FALLBACK_KEYS.items():
            if not isinstance(fallbacks.get(state, {}).get(key), typ):
                problems.append(f"fallbacks.{state}.{key} must be {typ.__name__}")
    if problems:
        raise DisposalMapError("Invalid disposal_map.json:\n  " + "\n  ".join(problems))


class DisposalEngine:
    """Turns a Prediction into a Recommendation using the validated disposal map."""

    def __init__(self, data: dict[str, Any], class_names: list[str]) -> None:
        validate_disposal_map(data, class_names)
        self.classes: dict[str, dict[str, Any]] = data["classes"]
        self.fallbacks: dict[str, dict[str, Any]] = data["fallbacks"]
        self.class_names = class_names

    @classmethod
    def from_file(cls, path: str | Path, class_names: list[str]) -> DisposalEngine:
        return cls(load_json(path), class_names)

    @classmethod
    def from_config(cls, config_path: str | Path, class_names: list[str]) -> DisposalEngine:
        return cls.from_file(load_yaml(config_path)["disposal_map"], class_names)

    def advice(self, label: str, probability: float = 1.0) -> ClassAdvice:
        """Guidance for one class."""
        entry = self.classes[label]
        return ClassAdvice(label=label, probability=probability,
                           **{k: entry[k] for k in REQUIRED_CLASS_KEYS})

    def recommend(self, prediction: Prediction) -> Recommendation:
        """Full guidance when confident, top-2 plus a safe default when uncertain,
        retake guidance only when unrecognised."""
        if prediction.state == State.CONFIDENT:
            primary = self.advice(prediction.label, prediction.confidence)
            return Recommendation(
                state=prediction.state,
                title=f"{primary.display_name}: use the {primary.bin} bin ({primary.bin_label})",
                primary=primary,
                candidates=[primary],
            )
        fb = self.fallbacks[prediction.state.value]
        candidates = (
            [self.advice(c, p) for c, p in prediction.top_k[:2]]
            if prediction.state == State.UNCERTAIN else []
        )
        return Recommendation(state=prediction.state, title=fb["title"], candidates=candidates,
                              rule=fb["rule"], advice=list(fb["advice"]))
