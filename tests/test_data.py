"""Data pipeline tests: class mapping, de-duplication, splitting, leakage, datasets, transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from tests.conftest import make_image
from waste.data.dataset import ManifestDataset, folder_frame
from waste.data.dedupe import deduplicate, near_duplicate_pairs, phash_int
from waste.data.prepare import find_dataset_root, scan_raw
from waste.data.split import UNUSED, cap_train, class_weights, cross_split_leaks, stratified_split
from waste.data.transforms import get_transform
from waste.utils import load_yaml, normalise_folder_name


def test_class_map_has_nine_unique_classes_and_sources():
    cmap = load_yaml("configs/class_map.yaml")["classes"]
    assert list(cmap) == ["paper", "cardboard", "plastic", "metal", "glass", "organic",
                          "battery", "textile", "general"]
    sources = [s for srcs in cmap.values() for s in srcs]
    assert len(sources) == len(set(sources)) == 12


def test_folder_names_normalise_across_mirrors():
    assert normalise_folder_name("Brown_Glass") == normalise_folder_name("brown-glass")
    assert normalise_folder_name("white glass ") == "white-glass"


def test_scan_raw_merges_and_skips(tmp_path: Path):
    for folder in ["brown-glass", "Green_Glass", "clothes", "unknown"]:
        (tmp_path / folder).mkdir()
        make_image(hash(folder) % 1000).save(tmp_path / folder / "a.jpg")
    cmap = load_yaml("configs/class_map.yaml")["classes"]
    df = scan_raw(tmp_path, cmap, [".jpg"])
    assert sorted(df["label"]) == ["glass", "glass", "textile"]


def test_find_dataset_root_locates_nested_folder(tmp_path: Path):
    nested = tmp_path / "archive" / "garbage_classification"
    for folder in ["paper", "battery", "trash"]:
        (nested / folder).mkdir(parents=True)
    assert find_dataset_root(tmp_path, {"paper", "battery", "trash"}) == nested


def test_near_duplicates_are_grouped_and_removed():
    base = make_image(1, 128)
    shifted = base.resize((127, 127)).resize((128, 128))  # visually identical copy
    hashes = [phash_int(base), phash_int(shifted), phash_int(make_image(2, 128))]
    df = pd.DataFrame({"path": ["a", "b", "c"], "label": ["paper", "paper", "metal"],
                       "hash": np.array(hashes, dtype=np.uint64)})
    clean, stats = deduplicate(df, max_distance=4)
    assert len(clean) == 2 and stats["removed_duplicates"] == 1


def test_conflicting_label_duplicates_are_dropped():
    h = phash_int(make_image(5, 128))
    df = pd.DataFrame({"path": ["a", "b"], "label": ["paper", "cardboard"],
                       "hash": np.array([h, h], dtype=np.uint64)})
    clean, stats = deduplicate(df, max_distance=0)
    assert clean.empty and stats["removed_label_conflicts"] == 2


def test_near_duplicate_pairs_hamming_distance():
    hashes = np.array([0b0000, 0b0011, 0b1111_0000], dtype=np.uint64)
    assert near_duplicate_pairs(hashes, 2) == [(0, 1)]


def _frame(n_per_class: dict[str, int]) -> pd.DataFrame:
    rows, rng = [], np.random.default_rng(0)
    for label, n in n_per_class.items():
        for i in range(n):
            # Random 64-bit hashes sit ~32 bits apart, so nothing counts as a duplicate.
            h = np.uint64(rng.integers(0, 2**63, dtype=np.int64))
            rows.append({"path": f"{label}/{i}.jpg", "label": label, "hash": h})
    return pd.DataFrame(rows)


def test_stratified_split_ratios_and_reproducibility():
    df = _frame({"paper": 100, "metal": 40})
    a = stratified_split(df, 0.7, 0.15, 0.15, seed=42)
    b = stratified_split(df, 0.7, 0.15, 0.15, seed=42)
    pd.testing.assert_frame_equal(a, b)
    counts = a.groupby(["label", "split"]).size()
    assert counts["paper", "train"] == 70 and counts["metal", "train"] == 28
    assert set(a["path"]) == set(df["path"])  # every image lands in exactly one split


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        stratified_split(_frame({"paper": 10}), 0.8, 0.15, 0.15, seed=1)


def test_cap_train_only_touches_training_rows():
    split = stratified_split(_frame({"textile": 200, "metal": 40}), 0.7, 0.15, 0.15, seed=42)
    capped = cap_train(split, max_per_class=50, seed=42)
    train = capped[capped["split"] == "train"]["label"].value_counts()
    assert train["textile"] == 50 and train["metal"] == 28
    assert (capped["split"] == UNUSED).sum() == 140 - 50
    for s in ("val", "test"):
        assert (capped["split"] == s).sum() == (split["split"] == s).sum()


def test_no_cross_split_leakage_detected_and_reported():
    df = stratified_split(_frame({"paper": 20, "metal": 20}), 0.7, 0.15, 0.15, seed=0)
    assert cross_split_leaks(df, max_distance=4) == 0
    leaked = df.copy()
    test_idx = leaked.index[leaked["split"] == "test"][0]
    train_idx = leaked.index[leaked["split"] == "train"][0]
    leaked.loc[test_idx, "hash"] = leaked.loc[train_idx, "hash"]
    assert cross_split_leaks(leaked, max_distance=4) == 1


def test_class_weights_inverse_frequency(class_names):
    labels = pd.Series(["paper"] * 30 + ["metal"] * 10)
    w = dict(zip(class_names, class_weights(labels, class_names), strict=True))
    assert w["metal"] == pytest.approx(3 * w["paper"])


def test_manifest_dataset_and_transforms(image_folder: Path, class_names):
    cmap = {c: [c] for c in class_names}
    frame = folder_frame(image_folder, cmap, [".png"])
    assert len(frame) == 12 * len(class_names)
    ds = ManifestDataset(frame, class_names, get_transform(train=True))
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and isinstance(y, int)
    x_eval, _ = ManifestDataset(frame, class_names, get_transform(train=False))[0]
    assert x_eval.shape == (3, 224, 224) and torch.isfinite(x_eval).all()


def test_dataset_rejects_unknown_labels(class_names):
    with pytest.raises(ValueError):
        ManifestDataset(pd.DataFrame({"path": ["x.jpg"], "label": ["styrofoam"]}), class_names)


def test_eval_transform_is_deterministic():
    img = Image.fromarray(np.full((300, 400, 3), 128, dtype=np.uint8))
    t = get_transform(train=False)
    assert torch.equal(t(img), t(img))
