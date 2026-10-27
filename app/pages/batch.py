"""Batch page (S3): many images or a ZIP -> results table -> CSV download."""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath

import common
import pandas as pd
import streamlit as st

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

st.title("Batch classification")
st.caption("Upload several photos or one ZIP archive. Each image gets a class, a confidence "
           "state and a bin.")

predictor, engine = common.require_model()
cfg = common.app_config()
store = common.get_store()
limit = cfg["app"]["max_batch_images"]

files = st.file_uploader("Images or a ZIP", type=["jpg", "jpeg", "png", "webp", "bmp", "zip"],
                         accept_multiple_files=True)
if not files:
    st.info(f"Add up to {limit} images (or ZIP archives of images) to start.")
    st.stop()


def collect(uploaded: list) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Flatten uploads and ZIP members into (name, bytes) pairs; returns skipped names too."""
    items, skipped = [], []
    for f in uploaded:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as zf:
                for info in zf.infolist():
                    name = PurePosixPath(info.filename)
                    if info.is_dir() or name.name.startswith(".") or "__MACOSX" in name.parts:
                        continue
                    if name.suffix.lower() in IMAGE_EXT:
                        items.append((f"{f.name}/{info.filename}", zf.read(info)))
        else:
            items.append((f.name, f.getvalue()))
    if len(items) > limit:
        skipped = [n for n, _ in items[limit:]]
        items = items[:limit]
    return items, skipped


run_key = "|".join(f"{f.name}:{f.size}" for f in files)
if st.session_state.get("batch_key") != run_key:
    items, over_limit = collect(files)
    images, names, unreadable = [], [], []
    for name, data in items:
        try:
            images.append(common.open_image(data))
            names.append(name)
        except Exception:
            unreadable.append(name)
    rows = []
    progress = st.progress(0.0, text="Classifying...")
    step = 16
    for i in range(0, len(images), step):
        for name, pred in zip(names[i : i + step], predictor.predict_batch(images[i : i + step]),
                              strict=True):
            rec = engine.recommend(pred)
            a = rec.primary
            store.log_prediction("batch", pred.label, pred.confidence, pred.state.value,
                                 a.bin if a else None, a.recyclable if a else None,
                                 predictor.model_name)
            rows.append({
                "file": name,
                "class": engine.classes[pred.label]["display_name"],
                "confidence": round(pred.confidence, 4),
                "state": pred.state.value,
                "bin": a.bin if a else "check manually",
                "second_choice": pred.top_k[1][0],
            })
        progress.progress(min((i + step) / max(len(images), 1), 1.0), text="Classifying...")
    progress.empty()
    st.session_state.update(batch_key=run_key, batch_rows=rows,
                            batch_skipped=over_limit, batch_unreadable=unreadable)

df = pd.DataFrame(st.session_state["batch_rows"])
if st.session_state["batch_skipped"]:
    st.warning(f"Only the first {limit} images were processed "
               f"({len(st.session_state['batch_skipped'])} skipped).")
if st.session_state["batch_unreadable"]:
    st.warning("Could not read: " + ", ".join(st.session_state["batch_unreadable"]))
if df.empty:
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Images", len(df))
c2.metric("Confident", f"{(df['state'] == 'confident').mean():.0%}")
c3.metric("Need checking", int((df["state"] != "confident").sum()))
c4.metric("Mean confidence", f"{df['confidence'].mean():.0%}")

st.dataframe(
    df,
    hide_index=True,
    width="stretch",
    column_config={
        "confidence": st.column_config.ProgressColumn("confidence", format="percent",
                                                      min_value=0, max_value=1),
        "second_choice": st.column_config.TextColumn("2nd choice"),
    },
)
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="waste_batch_results.csv", mime="text/csv", type="primary")

by_bin = df["bin"].value_counts().rename_axis("bin").reset_index(name="items")
st.altair_chart(common.bar_chart(by_bin, "bin", "items", "Items per bin"), width="stretch")
