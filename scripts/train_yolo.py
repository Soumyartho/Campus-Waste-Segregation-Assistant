"""Optional C1: YOLOv8n-cls as a YOLO-family comparison model.

Builds an ultralytics folder dataset from the same manifest (hard links, no copies where
possible), trains yolov8n-cls, evaluates on the test split, and writes
reports/yolov8n_cls/eval_test/metrics.json in the same schema as waste.evaluate, so
'python -m waste.evaluate compare' includes it.

Requires: pip install ultralytics
Usage: python scripts/train_yolo.py [--epochs 20] [--imgsz 224]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste.data.dataset import read_manifest  # noqa: E402
from waste.evaluate import compute_metrics, prediction_frame  # noqa: E402
from waste.utils import class_names_from_map, ensure_dir, load_yaml, resolve_path, save_json  # noqa: E402

RUN = "yolov8n_cls"


def build_dataset(out: Path) -> None:
    """Create out/{train,val,test}/<class>/ image links from the manifest."""
    cfg = load_yaml("configs/data.yaml")
    manifest = read_manifest(cfg["paths"]["manifest"])
    for row in manifest[manifest["split"].isin(["train", "val", "test"])].itertuples():
        dst = out / row.split / row.label / Path(row.path).name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(resolve_path(row.path), dst)
        except OSError:
            shutil.copy2(resolve_path(row.path), dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed: pip install ultralytics") from exc

    data_dir = resolve_path("data/yolo_cls")
    build_dataset(data_dir)
    model = YOLO("yolov8n-cls.pt")
    t0 = time.time()
    model.train(data=str(data_dir), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                seed=42, patience=4, project=str(resolve_path("models")), name=RUN,
                exist_ok=True, verbose=False)
    best = resolve_path(f"models/{RUN}/weights/best.pt")
    model = YOLO(str(best))

    class_names = class_names_from_map("configs/class_map.yaml")
    manifest = read_manifest(load_yaml("configs/data.yaml")["paths"]["manifest"])
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    yolo_names = model.names  # index -> folder name (alphabetical)
    order = [class_names.index(yolo_names[i]) for i in range(len(yolo_names))]
    probs = np.zeros((len(test), len(class_names)))
    for start in range(0, len(test), 64):
        paths = [str(resolve_path(p)) for p in test["path"][start : start + 64]]
        for k, res in enumerate(model.predict(paths, imgsz=args.imgsz, verbose=False)):
            probs[start + k, order] = res.probs.data.cpu().numpy()
    targets = test["label"].map(class_names.index).to_numpy()

    out = ensure_dir(Path("reports") / RUN / "eval_test")
    metrics = compute_metrics(probs, targets, class_names)
    prediction_frame(test, probs, class_names).to_csv(out / "predictions.csv", index=False)
    x = str(resolve_path(test["path"][0]))
    times = []
    for _ in range(25):
        s = time.perf_counter()
        model.predict(x, imgsz=args.imgsz, device="cpu", verbose=False)
        times.append((time.perf_counter() - s) * 1000)
    metrics.update(run_name=RUN, model="yolov8n-cls", split="test",
                   size_mb=round(best.stat().st_size / 1e6, 2),
                   cpu_latency_ms=round(float(np.median(times[5:])), 2),
                   best_val_macro_f1=None, best_epoch=None, git_commit=None,
                   train_minutes=round((time.time() - t0) / 60, 1))
    save_json(metrics, out / "metrics.json")
    print(f"YOLOv8n-cls test acc {metrics['accuracy']:.4f} macro-F1 {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
