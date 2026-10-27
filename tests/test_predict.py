"""Inference tests: state thresholds, top-k, the Predictor on a real checkpoint, logging,
and the evaluation helpers that tune the threshold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_image
from waste.checkpoint import save_checkpoint
from waste.evaluate import (
    apply_temperature,
    choose_tau,
    compute_metrics,
    coverage_curve,
    expected_calibration_error,
)
from waste.logstore import LogStore
from waste.models.factory import build_model
from waste.predict import Predictor, State, classify_state, prediction_from_probs


@pytest.mark.parametrize(
    ("p", "state"),
    [(0.95, State.CONFIDENT), (0.60, State.CONFIDENT), (0.59, State.UNCERTAIN),
     (0.35, State.UNCERTAIN), (0.34, State.UNRECOGNISED), (0.0, State.UNRECOGNISED)],
)
def test_state_boundaries(p, state):
    assert classify_state(p, tau=0.60, reject=0.35) == state


def test_top_k_sorted_and_complete(class_names):
    probs = np.linspace(0.01, 0.2, len(class_names))
    probs /= probs.sum()
    pred = prediction_from_probs(probs, class_names, 0.6, 0.35, top_k=3)
    assert [c for c, _ in pred.top_k] == class_names[::-1][:3]
    assert pred.top_k[0][1] >= pred.top_k[1][1] >= pred.top_k[2][1]
    assert sum(pred.probabilities.values()) == pytest.approx(1.0)


@pytest.fixture
def predictor(tmp_path: Path, class_names) -> Predictor:
    model = build_model("custom_cnn", len(class_names)).eval()
    save_checkpoint(tmp_path / "best.pt", model, "custom_cnn", {"dropout": 0.4}, class_names,
                    {"run_name": "tiny"}, {})
    from waste.checkpoint import load_checkpoint

    m, meta = load_checkpoint(tmp_path / "best.pt")
    return Predictor(m, meta, tau=0.6, reject=0.35)


def test_predictor_returns_valid_prediction(predictor, class_names):
    preds = predictor.predict_batch([make_image(i, 300) for i in range(3)])
    assert len(preds) == 3
    for p in preds:
        assert p.label in class_names and 0 <= p.confidence <= 1
        assert len(p.top_k) == 3 and isinstance(p.state, State)
    assert predictor.model_name == "tiny"


def test_temperature_flattens_probabilities(predictor):
    img = make_image(7, 256)
    sharp = predictor.probabilities([img])[0].max()
    predictor.temperature = 5.0
    assert predictor.probabilities([img])[0].max() <= sharp


def test_predictor_rejects_bad_thresholds(predictor):
    with pytest.raises(ValueError):
        Predictor(predictor.model, predictor.meta, tau=0.3, reject=0.5)


def test_missing_checkpoint_raises(tmp_path: Path):
    cfg = tmp_path / "inference.yaml"
    cfg.write_text(f"model: {{run_name: none, models_dir: '{tmp_path.as_posix()}', device: cpu}}\n"
                   "thresholds: {tau: 0.6, reject: 0.35, top_k: 3}\n")
    with pytest.raises(FileNotFoundError):
        Predictor.from_config(cfg)


def test_logstore_records_predictions_and_feedback(tmp_path: Path):
    store = LogStore(tmp_path / "app.db")
    pid = store.log_prediction("upload", "plastic", 0.91, "confident", "Blue", True, "m")
    store.log_feedback(pid, "plastic", "glass", make_image(1), tmp_path / "fb")
    df = store.predictions()
    assert len(df) == 1 and df.loc[0, "predicted_class"] == "plastic"
    fb = store.feedback()
    assert fb.loc[0, "correct_class"] == "glass" and Path(fb.loc[0, "image_path"]).exists()


def test_metrics_and_coverage_helpers(class_names):
    rng = np.random.default_rng(0)
    targets = rng.integers(0, len(class_names), 200)
    probs = rng.dirichlet(np.ones(len(class_names)), 200)
    probs[np.arange(150), targets[:150]] += 3  # 150 confident correct predictions
    probs /= probs.sum(1, keepdims=True)
    m = compute_metrics(probs, targets, class_names)
    assert 0.7 < m["accuracy"] < 0.9 and m["top3_accuracy"] >= m["accuracy"]
    curve = coverage_curve(probs, targets)
    assert curve["coverage"].is_monotonic_decreasing
    tau = choose_tau(curve, target_accuracy=0.95, reject=0.35)
    assert tau >= 0.35
    assert curve.loc[curve["tau"] == tau, "accuracy"].iloc[0] >= 0.95
    assert 0 <= expected_calibration_error(probs, targets) <= 1
    assert np.allclose(apply_temperature(probs, 1.0), probs)
