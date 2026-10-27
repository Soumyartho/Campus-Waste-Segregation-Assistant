"""Export a trained checkpoint to ONNX and check it matches PyTorch (C3).

Usage: python scripts/export_onnx.py --run mobilenet_v2
Writes models/<run>/model.onnx.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste.checkpoint import checkpoint_path, load_checkpoint  # noqa: E402
from waste.utils import load_yaml  # noqa: E402


def export(run: str, opset: int = 17) -> Path:
    """Export models/<run>/best.pt to models/<run>/model.onnx with a dynamic batch axis."""
    models_dir = load_yaml("configs/inference.yaml")["model"]["models_dir"]
    ckpt = checkpoint_path(models_dir, run)
    model, meta = load_checkpoint(ckpt, "cpu")
    out = ckpt.with_name("model.onnx")
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, out, input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
        dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, 224, 224)
    with torch.inference_mode():
        ref = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"Exported {out} ({out.stat().st_size / 1e6:.1f} MB); max |torch - onnx| = {diff:.2e}")
    print(f"Classes: {meta['class_names']}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="mobilenet_v2")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.run, args.opset)
