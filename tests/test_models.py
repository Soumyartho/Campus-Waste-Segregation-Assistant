"""Model tests: output shapes, freezing phases, checkpoint round trip, a tiny training step."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from waste.checkpoint import load_checkpoint, save_checkpoint
from waste.models.factory import build_model, set_trainable, train_mode
from waste.train import make_scheduler

N_CLASSES = 9


@pytest.mark.parametrize("name", ["custom_cnn", "mobilenet_v2", "resnet18"])
def test_forward_shape(name):
    model = build_model(name, N_CLASSES, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, N_CLASSES)


def test_custom_cnn_matches_plan_architecture():
    model = build_model("custom_cnn", N_CLASSES)
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert [c.out_channels for c in convs] == [32, 64, 128, 256]
    assert model.classifier[0].p == 0.4 and model.classifier[1].in_features == 256


def test_mobilenet_head_matches_plan():
    model = build_model("mobilenet_v2", N_CLASSES, pretrained=False, dropout=0.3)
    assert model.classifier[0].p == 0.3
    assert (model.classifier[1].in_features, model.classifier[1].out_features) == (1280, 9)


def test_phase_freezing_mobilenet():
    model = build_model("mobilenet_v2", N_CLASSES, pretrained=False)
    head_only = set_trainable(model, "mobilenet_v2", "head")
    assert head_only == 1280 * N_CLASSES + N_CLASSES
    assert not any(p.requires_grad for p in model.features.parameters())
    partial = set_trainable(model, "mobilenet_v2", "last_blocks", unfreeze_blocks=5)
    assert head_only < partial < sum(p.numel() for p in model.parameters())
    assert all(p.requires_grad for p in model.features[-1].parameters())
    assert not any(p.requires_grad for p in model.features[0].parameters())


def test_frozen_batchnorm_stays_in_eval_mode():
    model = build_model("mobilenet_v2", N_CLASSES, pretrained=False)
    set_trainable(model, "mobilenet_v2", "head")
    train_mode(model)
    assert not model.features[0][1].training  # frozen BatchNorm
    assert model.classifier.training


def test_cosine_scheduler_decays():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-3)
    sched = make_scheduler(opt, "cosine", epochs=4)
    for _ in range(4):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] < 1e-6
    assert make_scheduler(opt, "none", 4) is None


def test_checkpoint_round_trip(tmp_path: Path, class_names):
    model = build_model("custom_cnn", N_CLASSES).eval()
    path = save_checkpoint(tmp_path / "best.pt", model, "custom_cnn", {"dropout": 0.4},
                           class_names, {"run_name": "t"}, {"val_macro_f1": 0.5})
    loaded, meta = load_checkpoint(path)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)
    assert meta["class_names"] == class_names and "git_commit" in meta


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    model = build_model("custom_cnn", N_CLASSES)
    x, y = torch.randn(8, 3, 64, 64), torch.randint(0, N_CLASSES, (8,))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
