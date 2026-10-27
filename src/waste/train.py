"""Generic training script for every model (PLAN.md Section 7.3).

Phases, learning rates, schedulers and early stopping all come from the YAML config.
Usage: python -m waste.train --config configs/train_mobilenet.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from waste.checkpoint import save_checkpoint
from waste.data.dataset import make_loader, split_frames
from waste.data.split import class_weights
from waste.evaluate import collect_outputs
from waste.models.factory import build_model, model_kwargs, set_trainable, train_mode
from waste.plotting import SERIES, apply_style, plt, save
from waste.utils import (
    class_names_from_map,
    ensure_dir,
    get_device,
    load_yaml,
    save_json,
    set_seed,
)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """One pass over the training data; returns (mean loss, accuracy)."""
    train_mode(model)
    total_loss, correct, seen = 0.0, 0, 0
    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def make_scheduler(
    optimizer: torch.optim.Optimizer, kind: str, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Per-epoch LR scheduler: 'cosine' decays to zero over the phase, 'none' keeps LR fixed."""
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if kind in (None, "none"):
        return None
    raise ValueError(f"Unknown scheduler '{kind}'")


def plot_history(history: pd.DataFrame, out_dir: Path, run_name: str) -> None:
    """Training/validation loss and accuracy curves with phase boundaries marked."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ep = history["epoch"]
    ax1.plot(ep, history["train_loss"], color=SERIES[0], label="train")
    ax1.plot(ep, history["val_loss"], color=SERIES[1], label="validation")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy (label-smoothed)")
    ax2.plot(ep, history["train_acc"], color=SERIES[0], label="train accuracy")
    ax2.plot(ep, history["val_acc"], color=SERIES[1], label="validation accuracy")
    ax2.plot(ep, history["val_macro_f1"], color=SERIES[2], label="validation macro-F1")
    ax2.set(title="Accuracy and macro-F1", xlabel="Epoch", ylabel="Score")
    best = history.loc[history["val_macro_f1"].idxmax()]
    ax2.scatter([best["epoch"]], [best["val_macro_f1"]], s=40, color=SERIES[2],
                edgecolor="white", linewidth=1.5, zorder=3)
    ax2.annotate(f"best {best['val_macro_f1']:.3f}", (best["epoch"], best["val_macro_f1"]),
                 textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8)
    for ax in (ax1, ax2):
        for boundary in history.loc[history["phase"].ne(history["phase"].shift()), "epoch"][1:]:
            ax.axvline(boundary - 0.5, color="#9a9893", linestyle="--", linewidth=1)
        ax.legend()
    fig.suptitle(run_name, x=0.01, ha="left", fontsize=10, color="#52514e")
    save(fig, out_dir / "curves.png")


def train(config_path: str, max_batches: int | None = None, num_workers: int | None = None) -> dict:
    """Train one run from its YAML config; returns the run summary."""
    cfg = load_yaml(config_path)
    tcfg, mcfg, run = cfg["train"], cfg["model"], cfg["run_name"]
    data_cfg = load_yaml(cfg["data_config"])
    set_seed(tcfg["seed"])
    device = get_device()
    use_amp = bool(tcfg.get("amp")) and device.type == "cuda"

    class_names = class_names_from_map(data_cfg["paths"]["class_map"])
    frames = split_frames(cfg["data_config"])
    workers = data_cfg["loader"]["num_workers"] if num_workers is None else num_workers
    bs = tcfg["batch_size"]
    train_loader = make_loader(frames["train"], class_names, True, bs, workers, cfg["augment"])
    val_loader = make_loader(frames["val"], class_names, False, bs, workers)

    model = build_model(mcfg["name"], len(class_names), **model_kwargs(mcfg)).to(device)
    weights = (
        class_weights(frames["train"]["label"], class_names) if tcfg["class_weights"] else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=device) if weights else None,
        label_smoothing=tcfg["label_smoothing"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    models_dir = ensure_dir(Path(cfg["output"]["models_dir"]) / run)
    reports_dir = ensure_dir(Path(cfg["output"]["reports_dir"]) / run)
    print(f"Run '{run}' on {device} | train {len(frames['train'])} | val {len(frames['val'])}")

    history: list[dict[str, Any]] = []
    best_f1, best_row, epoch, start = -1.0, None, 0, time.time()
    patience = tcfg["early_stopping"]["patience"]
    for phase in tcfg["phases"]:
        n_trainable = set_trainable(
            model, mcfg["name"], phase["trainable"], phase.get("unfreeze_blocks", 0)
        )
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=phase["lr"], weight_decay=tcfg["weight_decay"])
        scheduler = make_scheduler(optimizer, phase.get("scheduler"), phase["epochs"])
        print(f"Phase '{phase['name']}': {n_trainable:,} trainable params, lr {phase['lr']}")
        stale = 0
        for _ in range(phase["epochs"]):
            epoch += 1
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, use_amp, max_batches
            )
            probs, targets, val_loss = collect_outputs(
                model, val_loader, device, criterion, max_batches
            )
            preds = probs.argmax(1)
            row = {
                "epoch": epoch, "phase": phase["name"], "lr": lr,
                "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss,
                "val_acc": float((preds == targets).mean()),
                "val_macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
                "seconds": round(time.time() - t0, 1),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(reports_dir / "history.csv", index=False)
            improved = row["val_macro_f1"] > best_f1
            print(
                f"  epoch {epoch:2d} [{phase['name']}] loss {tr_loss:.3f}/{val_loss:.3f} "
                f"acc {tr_acc:.3f}/{row['val_acc']:.3f} val-F1 {row['val_macro_f1']:.4f} "
                f"({row['seconds']}s){'  *best*' if improved else ''}"
            )
            if improved:
                best_f1, best_row, stale = row["val_macro_f1"], row, 0
                save_checkpoint(models_dir / "best.pt", model, mcfg["name"], model_kwargs(mcfg),
                                class_names, cfg, {k: row[k] for k in row if k != "lr"})
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step()
            if stale >= patience:
                print(f"  early stopping: no val macro-F1 gain for {patience} epochs")
                break

    hist = pd.DataFrame(history)
    plot_history(hist, reports_dir, run)
    summary = {
        "run_name": run, "model": mcfg["name"], "device": str(device), "amp": use_amp,
        "epochs_run": epoch, "best_epoch": best_row["epoch"], "best_phase": best_row["phase"],
        "best_val_macro_f1": best_f1, "best_val_acc": best_row["val_acc"],
        "train_minutes": round((time.time() - start) / 60, 2),
        "train_images": len(frames["train"]), "class_weights": weights,
        "checkpoint": str(models_dir / "best.pt"),
    }
    save_json(summary, reports_dir / "train_summary.json")
    print(f"Best val macro-F1 {best_f1:.4f} at epoch {best_row['epoch']} "
          f"-> {models_dir / 'best.pt'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="limit batches per epoch (smoke tests only)")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.max_batches, args.num_workers)
