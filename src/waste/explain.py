"""Grad-CAM explanations (S1): heatmap overlays for the app and an error gallery for the report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import transforms as T

from waste.data.dataset import load_image
from waste.data.transforms import IMAGE_SIZE, RESIZE, eval_transform
from waste.models.factory import gradcam_target_layer
from waste.plotting import apply_style, plt, save

_display_crop = T.Compose([T.Resize(RESIZE), T.CenterCrop(IMAGE_SIZE)])


def gradcam_overlay(
    model: nn.Module, model_name: str, image: Image.Image, target_idx: int | None = None
) -> np.ndarray:
    """Return an RGB uint8 overlay of the Grad-CAM heatmap for ``target_idx`` (default: top class).

    The overlay matches the 224x224 centre crop the model actually sees.
    """
    image = image.convert("RGB")
    x = eval_transform()(image).unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    targets = None if target_idx is None else [ClassifierOutputTarget(int(target_idx))]
    with GradCAM(model=model, target_layers=[gradcam_target_layer(model, model_name)]) as cam:
        heat = cam(input_tensor=x, targets=targets)[0]
    rgb = np.asarray(_display_crop(image), dtype=np.float32) / 255.0
    return show_cam_on_image(rgb, heat, use_rgb=True, image_weight=0.55)


def top_confusions(preds: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """The ``k`` most frequent (true, predicted) error pairs from a predictions frame."""
    errors = preds[preds["true"] != preds["pred"]]
    return (
        errors.groupby(["true", "pred"]).size().rename("count").reset_index()
        .sort_values("count", ascending=False).head(k).reset_index(drop=True)
    )


def error_gallery(
    model: nn.Module,
    model_name: str,
    preds: pd.DataFrame,
    class_names: list[str],
    out_path: str | Path,
    pairs: int = 5,
    per_pair: int = 3,
) -> Path | None:
    """Grid of the most common confusions: each example shows the image and its Grad-CAM."""
    conf = top_confusions(preds, pairs)
    if conf.empty:
        return None
    apply_style()
    fig, axes = plt.subplots(len(conf), per_pair * 2, figsize=(per_pair * 2 * 1.7, len(conf) * 2.0))
    axes = np.atleast_2d(axes)
    for r, row in conf.iterrows():
        sel = preds[(preds["true"] == row["true"]) & (preds["pred"] == row["pred"])]
        sel = sel.sort_values("confidence", ascending=False).head(per_pair)
        for k in range(per_pair):
            a_img, a_cam = axes[r, 2 * k], axes[r, 2 * k + 1]
            a_img.axis("off")
            a_cam.axis("off")
            if k >= len(sel):
                continue
            ex = sel.iloc[k]
            img = load_image(ex["path"])
            a_img.imshow(_display_crop(img))
            a_cam.imshow(gradcam_overlay(model, model_name, img, class_names.index(ex["pred"])))
            a_img.set_title(f"p={ex['confidence']:.2f}", fontsize=7)
        axes[r, 0].text(-0.1, 0.5, f"{row['true']}\n-> {row['pred']}\n({row['count']}x)",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle("Most common confusions (true -> predicted): image and Grad-CAM for the "
                 "predicted class", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, out_path, dpi=120)
