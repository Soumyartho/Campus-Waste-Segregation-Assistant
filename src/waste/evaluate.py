"""Evaluation suite (PLAN.md Section 8).

Sub-commands:
  run       evaluate one run on a manifest split           --run mobilenet_v2 --split test
  all       evaluate every trained run on a split          --split test
  external  evaluate one run on TrashNet or the campus set --run mobilenet_v2 --dataset campus
  coverage  accuracy-vs-coverage curve, temperature and tau on validation  --run mobilenet_v2
  compare   collect all metrics into reports/comparison.md
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch import nn

from waste.plotting import SEQUENTIAL, SERIES, apply_style, plt, save
from waste.utils import ensure_dir, load_yaml, resolve_path, save_json

DATA_CONFIG = "configs/data.yaml"
INFERENCE_CONFIG = "configs/inference.yaml"


@torch.inference_mode()
def collect_outputs(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a model over a loader; returns (softmax probabilities, targets, mean loss)."""
    model.eval()
    probs, targets, total, seen = [], [], 0.0, 0
    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        if criterion is not None:
            total += criterion(logits, y).item() * len(y)
        seen += len(y)
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        targets.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(targets), total / max(seen, 1)


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    """ECE: confidence-weighted gap between confidence and accuracy across equal-width bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == targets
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(probs: np.ndarray, targets: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy, macro-F1, top-3 and per-class report over the classes present in ``targets``."""
    preds = probs.argmax(1)
    present = sorted(set(targets.tolist()))
    k = min(3, len(class_names))
    report = classification_report(
        targets, preds, labels=present, target_names=[class_names[i] for i in present],
        output_dict=True, zero_division=0,
    )
    return {
        "n": int(len(targets)),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, labels=present, average="macro",
                                   zero_division=0)),
        "top3_accuracy": float(top_k_accuracy_score(targets, probs, k=k,
                                                    labels=list(range(len(class_names))))),
        "ece": expected_calibration_error(probs, targets),
        "classes_evaluated": [class_names[i] for i in present],
        "per_class": {c: report[c] for c in (class_names[i] for i in present)},
    }


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path, title: str, norm: bool) -> None:
    """Heatmap of a confusion matrix; ``norm`` shows row-normalised recall per true class."""
    apply_style()
    data = cm / np.clip(cm.sum(1, keepdims=True), 1, None) if norm else cm
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(
        data, annot=True, fmt=".2f" if norm else "d", cmap=SEQUENTIAL, cbar=False, ax=ax,
        xticklabels=labels, yticklabels=labels, linewidths=1, linecolor="#fcfcfb",
        annot_kws={"fontsize": 8}, vmin=0, vmax=1 if norm else None,
    )
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save(fig, path)


def cpu_latency_ms(model: nn.Module, runs: int = 30, threads: int | None = None) -> float:
    """Median single-image CPU latency in milliseconds, optionally with a fixed thread count."""
    previous = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, 224, 224)
    times = []
    with torch.inference_mode():
        for i in range(runs + 5):
            t0 = time.perf_counter()
            model(x)
            if i >= 5:
                times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(previous)
    return float(np.median(times))


def prediction_frame(
    frame: pd.DataFrame, probs: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    """Per-image predictions with confidence and top-3 classes."""
    top3 = np.argsort(-probs, 1)[:, :3]
    return pd.DataFrame(
        {
            "path": frame["path"].to_numpy(),
            "true": frame["label"].to_numpy(),
            "pred": [class_names[i] for i in probs.argmax(1)],
            "confidence": probs.max(1).round(4),
            "top3": ["|".join(class_names[j] for j in row) for row in top3],
        }
    )


def _latency_threads() -> int | None:
    return load_yaml(INFERENCE_CONFIG).get("benchmark", {}).get("cpu_threads")


def _load_run(run: str, device: torch.device) -> tuple[nn.Module, dict]:
    from waste.checkpoint import checkpoint_path, load_checkpoint

    models_dir = load_yaml(INFERENCE_CONFIG)["model"]["models_dir"]
    return load_checkpoint(checkpoint_path(models_dir, run), device)


def evaluate_frame(run: str, frame: pd.DataFrame, split_name: str, gallery: bool = True) -> dict:
    """Evaluate a run on a path/label frame; artefacts go to reports/<run>/eval_<split>/."""
    from waste.data.dataset import make_loader
    from waste.explain import error_gallery, top_confusions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = _load_run(run, device)
    class_names = meta["class_names"]
    workers = load_yaml(DATA_CONFIG)["loader"]["num_workers"]
    loader = make_loader(frame, class_names, False, 64, workers)
    probs, targets, _ = collect_outputs(model, loader, device)
    del loader  # release the worker processes so they do not skew the latency measurement

    out = ensure_dir(Path("reports") / run / f"eval_{split_name}")
    metrics = compute_metrics(probs, targets, class_names)
    preds = prediction_frame(frame, probs, class_names)
    preds.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(metrics["per_class"]).T.round(4).rename_axis("class").to_csv(
        out / "classification_report.csv"
    )
    present = sorted(set(targets.tolist()) | set(probs.argmax(1).tolist()))
    labels = [class_names[i] for i in present]
    cm = confusion_matrix(targets, probs.argmax(1), labels=present)
    plot_confusion(cm, labels, out / "confusion_matrix.png",
                   f"{run} - {split_name} (counts)", False)
    plot_confusion(cm, labels, out / "confusion_matrix_normalized.png",
                   f"{run} - {split_name} (row-normalised)", True)
    top_confusions(preds).to_csv(out / "top_confusions.csv", index=False)
    if gallery:
        error_gallery(model.to(device), meta["model_name"], preds, class_names,
                      out / "error_gallery.png")

    metrics.update(
        run_name=run, model=meta["model_name"], split=split_name,
        size_mb=round(meta["size_mb"], 2),
        cpu_latency_ms=round(cpu_latency_ms(model, threads=_latency_threads()), 2),
        best_val_macro_f1=meta["metrics"].get("val_macro_f1"),
        best_epoch=meta["metrics"].get("epoch"), git_commit=meta["git_commit"],
    )
    save_json(metrics, out / "metrics.json")
    print(f"[{run} | {split_name}] n={metrics['n']} acc={metrics['accuracy']:.4f} "
          f"macro-F1={metrics['macro_f1']:.4f} top3={metrics['top3_accuracy']:.4f} "
          f"ECE={metrics['ece']:.4f} latency={metrics['cpu_latency_ms']}ms -> {out}")
    return metrics


def evaluate_split(run: str, split: str) -> dict:
    """Evaluate a run on a manifest split (test for final numbers, val for decisions)."""
    from waste.data.dataset import split_frames

    return evaluate_frame(run, split_frames(DATA_CONFIG)[split], split)


def evaluate_external(run: str, dataset: str) -> dict | None:
    """Evaluate on TrashNet or the campus set using the external mappings in class_map.yaml."""
    from waste.data.dataset import folder_frame

    data_cfg = load_yaml(DATA_CONFIG)
    mapping = load_yaml(data_cfg["paths"]["class_map"])["external"][dataset]
    root = data_cfg["source"]["trashnet_dir" if dataset == "trashnet" else "campus_dir"]
    root_path = resolve_path(root)
    if dataset == "trashnet" and root_path.exists():
        # The archive nests the class folders one or two levels down.
        from waste.data.prepare import find_dataset_root
        from waste.utils import normalise_folder_name

        sources = {normalise_folder_name(s) for v in mapping.values() for s in v}
        root_path = find_dataset_root(root_path, sources)
    frame = folder_frame(root_path, mapping, data_cfg["image_extensions"])
    if frame.empty:
        print(f"No images found for '{dataset}' under {root_path}; skipping.")
        return None
    return evaluate_frame(run, frame, dataset)


def fit_temperature(probs: np.ndarray, targets: np.ndarray) -> float:
    """Temperature that minimises validation NLL (log-probabilities stand in for logits)."""
    logits = torch.log(torch.tensor(probs, dtype=torch.float64).clamp_min(1e-12))
    y = torch.tensor(targets)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities as softmax(log(p) / T)."""
    z = np.log(np.clip(probs, 1e-12, None)) / temperature
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def coverage_curve(probs: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """Coverage and selective accuracy for thresholds 0.00 to 0.99."""
    conf, correct = probs.max(1), probs.argmax(1) == targets
    rows = []
    for tau in np.round(np.arange(0, 1.0, 0.01), 2):
        kept = conf >= tau
        rows.append({"tau": tau, "coverage": kept.mean(),
                     "accuracy": correct[kept].mean() if kept.any() else np.nan})
    return pd.DataFrame(rows)


def choose_tau(curve: pd.DataFrame, target_accuracy: float, reject: float) -> float:
    """Lowest tau (max coverage) whose selective accuracy reaches the target; never below reject."""
    ok = curve[(curve["accuracy"] >= target_accuracy) & (curve["tau"] >= reject)]
    return float(ok["tau"].min()) if not ok.empty else float(curve["tau"].max())


def run_coverage(run: str, write_config: bool) -> dict:
    """S7 + C2: calibrate on validation, plot accuracy vs coverage and recommend tau."""
    from waste.data.dataset import make_loader, split_frames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = _load_run(run, device)
    inf = load_yaml(INFERENCE_CONFIG)
    target = inf["tuning"]["target_accuracy"]
    reject = inf["thresholds"]["reject"]
    val = split_frames(DATA_CONFIG)["val"]
    loader = make_loader(val, meta["class_names"], False, 64,
                         load_yaml(DATA_CONFIG)["loader"]["num_workers"])
    raw, targets, _ = collect_outputs(model, loader, device)
    temperature = fit_temperature(raw, targets)
    probs = apply_temperature(raw, temperature)
    curve = coverage_curve(probs, targets)
    tau = choose_tau(curve, target, reject)

    out = ensure_dir(Path("reports") / run)
    curve.to_csv(out / "coverage_curve.csv", index=False)
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(curve["coverage"], curve["accuracy"], color=SERIES[0], label="validation")
    sel = curve[curve["tau"] == tau].iloc[0]
    ax.scatter([sel["coverage"]], [sel["accuracy"]], s=50, color=SERIES[1], zorder=3,
               edgecolor="white", linewidth=1.5, label=f"chosen tau = {tau:.2f}")
    ax.annotate(f"tau {tau:.2f}: {sel['coverage']:.0%} answered, {sel['accuracy']:.1%} correct",
                (sel["coverage"], sel["accuracy"]), textcoords="offset points",
                xytext=(-10, -18), ha="right", fontsize=8)
    ax.set(xlabel="Coverage (share of images answered confidently)",
           ylabel="Accuracy on answered images", title=f"Accuracy vs coverage - {run}")
    ax.legend(loc="lower left")
    save(fig, out / "coverage_curve.png")

    result = {
        "run_name": run, "temperature": round(temperature, 4), "tau": tau, "reject": reject,
        "target_accuracy": target, "coverage_at_tau": float(sel["coverage"]),
        "accuracy_at_tau": float(sel["accuracy"]),
        "ece_before": expected_calibration_error(raw, targets),
        "ece_after": expected_calibration_error(probs, targets),
    }
    save_json(result, out / "threshold.json")
    print(json.dumps(result, indent=2))
    if write_config:
        _write_inference_values(tau, temperature)
        print(f"Updated {INFERENCE_CONFIG}: tau={tau}, temperature={temperature:.4f}")
    return result


def _write_inference_values(tau: float, temperature: float) -> None:
    """Replace the tau and temperature values in inference.yaml, keeping its comments."""
    p = resolve_path(INFERENCE_CONFIG)
    text = p.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^(\s*tau:\s*)[0-9.]+", rf"\g<1>{tau:.2f}", text)
    text = re.sub(r"(?m)^(\s*temperature:\s*)[0-9.]+", rf"\g<1>{temperature:.4f}", text)
    p.write_text(text, encoding="utf-8")


def compare() -> Path:
    """Write reports/comparison.md and comparison.csv from every metrics.json."""
    rows: list[dict[str, Any]] = []
    for f in sorted(resolve_path("reports").glob("*/eval_*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append({k: m.get(k) for k in (
            "run_name", "model", "split", "n", "accuracy", "macro_f1", "top3_accuracy", "ece",
            "size_mb", "cpu_latency_ms", "best_val_macro_f1", "best_epoch")})
    if not rows:
        raise SystemExit("No metrics found. Run 'python -m waste.evaluate all' first.")
    df = pd.DataFrame(rows)
    df.to_csv(resolve_path("reports/comparison.csv"), index=False)

    def fmt(x: Any, pct: bool = True) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{x:.1%}" if pct else f"{x:g}"

    lines = ["# Model comparison", "",
             "Generated by `python -m waste.evaluate compare`. Decisions use validation; "
             "test, TrashNet and campus numbers are final reports only.", "",
             "## Test split (Kaggle)", "",
             "| Run | Model | Accuracy | Macro-F1 | Top-3 | ECE | Size (MB) | CPU latency (ms) "
             "| Best val macro-F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in df[df["split"] == "test"].sort_values("macro_f1", ascending=False).itertuples():
        lines.append(f"| {r.run_name} | {r.model} | {fmt(r.accuracy)} | {fmt(r.macro_f1)} | "
                     f"{fmt(r.top3_accuracy)} | {r.ece:.3f} | {r.size_mb} | {r.cpu_latency_ms} | "
                     f"{fmt(r.best_val_macro_f1)} |")

    pivot = df.pivot_table(index="run_name", columns="split", values="accuracy")
    cols = [c for c in ("test", "trashnet", "campus") if c in pivot.columns]
    if len(cols) > 1:
        lines += ["", "## Domain shift (accuracy)", "",
                  "| Run | " + " | ".join({"test": "Kaggle test", "trashnet": "TrashNet",
                                           "campus": "Campus"}[c] for c in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for run, r in pivot[cols].iterrows():
            lines.append(f"| {run} | " + " | ".join(fmt(r[c]) for c in cols) + " |")

    test = df[df["split"] == "test"].set_index("run_name")
    ablations = [("Transfer learning", "custom_cnn", "mobilenet_v2"),
                 ("Augmentation", "mobilenet_v2_noaug", "mobilenet_v2"),
                 ("Fine-tuning", "mobilenet_v2_headonly", "mobilenet_v2")]
    abl = [(q, a, b) for q, a, b in ablations if a in test.index and b in test.index]
    if abl:
        lines += ["", "## Ablations (test macro-F1)", "",
                  "| Question | Without | With | Gain |", "|---|---|---|---|"]
        for q, a, b in abl:
            fa, fb = test.loc[a, "macro_f1"], test.loc[b, "macro_f1"]
            lines.append(f"| {q} | {a}: {fmt(fa)} | {b}: {fmt(fb)} | {(fb - fa) * 100:+.1f} pts |")

    out = resolve_path("reports/comparison.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out


def trained_runs() -> list[str]:
    """Run names that have a best.pt checkpoint."""
    models_dir = resolve_path(load_yaml(INFERENCE_CONFIG)["model"]["models_dir"])
    return sorted(p.parent.name for p in models_dir.glob("*/best.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p = sub.add_parser("all")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p = sub.add_parser("external")
    p.add_argument("--run", required=True, help="run name, or 'all'")
    p.add_argument("--dataset", required=True, choices=["trashnet", "campus"])
    p = sub.add_parser("coverage")
    p.add_argument("--run", required=True)
    p.add_argument("--write-config", action="store_true",
                   help="write the tuned tau and temperature into configs/inference.yaml")
    sub.add_parser("compare")
    args = parser.parse_args()

    if args.cmd == "run":
        evaluate_split(args.run, args.split)
    elif args.cmd == "all":
        for run in trained_runs():
            evaluate_split(run, args.split)
    elif args.cmd == "external":
        for run in trained_runs() if args.run == "all" else [args.run]:
            evaluate_external(run, args.dataset)
    elif args.cmd == "coverage":
        run_coverage(args.run, args.write_config)
    else:
        compare()


if __name__ == "__main__":
    main()
