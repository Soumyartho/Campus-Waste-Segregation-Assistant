"""CPU latency benchmark for every trained run, in PyTorch and (if exported) ONNX Runtime (C3).

Usage: python scripts/benchmark.py [--runs 50] [--threads 4]
Writes reports/benchmark.csv and reports/benchmark.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste.checkpoint import load_checkpoint  # noqa: E402
from waste.utils import load_yaml, resolve_path  # noqa: E402


def timed(fn, runs: int, warmup: int = 5) -> tuple[float, float]:
    """Median and 95th-percentile latency of ``fn`` in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--threads", type=int,
                        default=load_yaml("configs/inference.yaml")["benchmark"]["cpu_threads"])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    models_dir = resolve_path(load_yaml("configs/inference.yaml")["model"]["models_dir"])
    x = torch.randn(1, 3, 224, 224)
    rows = []
    for ckpt in sorted(models_dir.glob("*/best.pt")):
        run = ckpt.parent.name
        model, meta = load_checkpoint(ckpt, "cpu")
        params = sum(p.numel() for p in model.parameters()) / 1e6
        with torch.inference_mode():
            med, p95 = timed(lambda m=model: m(x), args.runs)
        rows.append({"run": run, "runtime": "pytorch", "params_m": round(params, 2),
                     "size_mb": round(meta["size_mb"], 2), "median_ms": round(med, 2),
                     "p95_ms": round(p95, 2)})
        onnx_path = ckpt.with_name("model.onnx")
        if onnx_path.exists():
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = args.threads
            sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
            xn = x.numpy()
            med, p95 = timed(lambda s=sess, xn=xn: s.run(None, {"image": xn}), args.runs)
            rows.append({"run": run, "runtime": "onnxruntime", "params_m": round(params, 2),
                         "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
                         "median_ms": round(med, 2), "p95_ms": round(p95, 2)})
        print(rows[-1])

    if not rows:
        raise SystemExit("No checkpoints found under models/. Train a model first.")
    df = pd.DataFrame(rows)
    df.to_csv(resolve_path("reports/benchmark.csv"), index=False)
    lines = [f"# CPU latency benchmark (batch 1, {args.threads} threads, {args.runs} runs)", "",
             "| Run | Runtime | Params (M) | Size (MB) | Median (ms) | p95 (ms) |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r.run} | {r.runtime} | {r.params_m} | {r.size_mb} | {r.median_ms} | "
              f"{r.p95_ms} |" for r in df.itertuples()]
    resolve_path("reports/benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
