"""Disposal engine tests: the three states, validation and the missing-class error."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from waste.disposal import DisposalEngine, DisposalMapError, validate_disposal_map
from waste.predict import State, prediction_from_probs
from waste.utils import load_json

MAP = load_json("configs/disposal_map.json")


def _probs(class_names, **values):
    p = np.full(len(class_names), 0.0)
    for k, v in values.items():
        p[class_names.index(k)] = v
    rest = (1 - p.sum()) / (p == 0).sum()
    p[p == 0] = rest
    return p


@pytest.fixture
def engine(class_names):
    return DisposalEngine(MAP, class_names)


def test_shipped_map_covers_every_class(class_names):
    validate_disposal_map(MAP, class_names)
    assert set(MAP["classes"]) == set(class_names)


def test_confident_returns_full_entry(engine, class_names):
    pred = prediction_from_probs(_probs(class_names, plastic=0.9), class_names, 0.6, 0.35)
    rec = engine.recommend(pred)
    assert rec.state == State.CONFIDENT
    assert rec.primary.label == "plastic" and rec.primary.bin == "Blue"
    assert rec.primary.probability == pytest.approx(0.9)
    assert rec.primary.steps and rec.primary.do_not and rec.rule is None


def test_uncertain_returns_top_two_and_safe_rule(engine, class_names):
    pred = prediction_from_probs(_probs(class_names, glass=0.45, plastic=0.3), class_names,
                                 0.6, 0.35)
    rec = engine.recommend(pred)
    assert rec.state == State.UNCERTAIN and rec.primary is None
    assert [c.label for c in rec.candidates] == ["glass", "plastic"]
    assert "when in doubt" in rec.rule.lower()


def test_unrecognised_returns_retake_guidance_only(engine, class_names):
    pred = prediction_from_probs(_probs(class_names, metal=0.2), class_names, 0.6, 0.35)
    rec = engine.recommend(pred)
    assert rec.state == State.UNRECOGNISED
    assert rec.primary is None and rec.candidates == []
    assert any("background" in a for a in rec.advice)


def test_missing_class_raises_clear_error(class_names):
    broken = copy.deepcopy(MAP)
    del broken["classes"]["battery"]
    with pytest.raises(DisposalMapError, match="battery"):
        DisposalEngine(broken, class_names)


def test_malformed_entry_raises(class_names):
    broken = copy.deepcopy(MAP)
    broken["classes"]["paper"]["steps"] = "not a list"
    with pytest.raises(DisposalMapError, match="paper.steps"):
        validate_disposal_map(broken, class_names)


def test_missing_fallback_raises(class_names):
    broken = copy.deepcopy(MAP)
    del broken["fallbacks"]["uncertain"]
    with pytest.raises(DisposalMapError, match="uncertain"):
        validate_disposal_map(broken, class_names)


def test_recommendation_serialises(engine, class_names):
    pred = prediction_from_probs(_probs(class_names, battery=0.95), class_names, 0.6, 0.35)
    d = engine.recommend(pred).to_dict()
    assert d["state"] == "confident" and d["primary"]["bin"] == "Red"
