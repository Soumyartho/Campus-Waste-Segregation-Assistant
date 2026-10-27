"""Model insights page: metrics, curves, confusion matrices and comparisons from reports/."""

from __future__ import annotations

import json
from pathlib import Path

import common
import pandas as pd
import streamlit as st

from waste.utils import resolve_path

st.title("Model insights")
st.caption("Everything here is read from `reports/`, produced by `make eval`. Test numbers are "
           "final results; all tuning used the validation split.")

reports = resolve_path(common.app_config()["app"]["reports_dir"])
runs = sorted(p.name for p in reports.iterdir() if p.is_dir() and (
    (p / "history.csv").exists() or any(p.glob("eval_*")))) if reports.exists() else []
if not runs:
    st.info("No reports yet. Train a model (`make train`) and evaluate it (`make eval`).")
    st.stop()


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def show_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"Not generated yet: `{path.relative_to(resolve_path('.'))}`")


comparison = reports / "comparison.csv"
if comparison.exists():
    st.subheader("Model comparison")
    cmp = pd.read_csv(comparison)
    test = cmp[cmp["split"] == "test"].sort_values("macro_f1", ascending=False)
    st.dataframe(
        test[["run_name", "model", "accuracy", "macro_f1", "top3_accuracy", "ece", "size_mb",
              "cpu_latency_ms"]],
        hide_index=True, width="stretch",
        column_config={
            "run_name": "Run", "model": "Model",
            "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
            "macro_f1": st.column_config.NumberColumn("Macro-F1", format="percent"),
            "top3_accuracy": st.column_config.NumberColumn("Top-3", format="percent"),
            "ece": st.column_config.NumberColumn("ECE", format="%.3f"),
            "size_mb": st.column_config.NumberColumn("Size (MB)", format="%.1f"),
            "cpu_latency_ms": st.column_config.NumberColumn("CPU ms / image", format="%.1f"),
        },
    )
    shift = cmp.pivot_table(index="run_name", columns="split", values="accuracy")
    cols = [c for c in ("test", "trashnet", "campus") if c in shift.columns]
    if len(cols) > 1:
        st.markdown("**Domain shift (accuracy)**")
        st.dataframe(shift[cols].rename_axis("Run").rename(
                         columns={"test": "Kaggle test", "trashnet": "TrashNet",
                                  "campus": "Campus"}),
                     width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent")
                                    for c in ("Kaggle test", "TrashNet", "Campus")})

default_run = common.app_config()["model"]["run_name"]
run = st.selectbox("Run", runs, index=runs.index(default_run) if default_run in runs else 0)
rdir = reports / run
evals = sorted(p.name.removeprefix("eval_") for p in rdir.glob("eval_*"))

metrics = read_json(rdir / "eval_test" / "metrics.json")
if metrics:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    k2.metric("Test macro-F1", f"{metrics['macro_f1']:.1%}")
    k3.metric("Top-3 accuracy", f"{metrics['top3_accuracy']:.1%}")
    k4.metric("CPU latency", f"{metrics['cpu_latency_ms']:.0f} ms")

tab_curves, tab_cm, tab_report, tab_thr, tab_err, tab_data = st.tabs(
    ["Training curves", "Confusion matrix", "Per-class report", "Threshold", "Errors", "Dataset"]
)
with tab_curves:
    show_image(rdir / "curves.png")
    summary = read_json(rdir / "train_summary.json")
    if summary:
        st.caption(f"Best validation macro-F1 {summary['best_val_macro_f1']:.3f} at epoch "
                   f"{summary['best_epoch']} ({summary['best_phase']} phase), "
                   f"{summary['train_minutes']} min on {summary['device']}.")

with tab_cm:
    split = st.radio("Evaluation set", evals or ["test"], horizontal=True, key="cm_split")
    norm, counts = st.tabs(["Row-normalised", "Counts"])
    with norm:
        show_image(rdir / f"eval_{split}" / "confusion_matrix_normalized.png")
    with counts:
        show_image(rdir / f"eval_{split}" / "confusion_matrix.png")

with tab_report:
    split = st.radio("Evaluation set", evals or ["test"], horizontal=True, key="rep_split")
    rep = rdir / f"eval_{split}" / "classification_report.csv"
    if rep.exists():
        st.dataframe(pd.read_csv(rep), hide_index=True, width="stretch")
    conf = rdir / f"eval_{split}" / "top_confusions.csv"
    if conf.exists():
        st.markdown("**Most common confusions**")
        st.dataframe(pd.read_csv(conf), hide_index=True, width="stretch")

with tab_thr:
    thr = read_json(rdir / "threshold.json")
    if thr:
        st.markdown(
            f"Chosen **tau = {thr['tau']:.2f}** (temperature {thr['temperature']:.2f}): "
            f"{thr['coverage_at_tau']:.0%} of validation images are answered confidently with "
            f"{thr['accuracy_at_tau']:.1%} accuracy (target {thr['target_accuracy']:.0%}). "
            f"Calibration error (ECE) {thr['ece_before']:.3f} -> {thr['ece_after']:.3f}."
        )
    show_image(rdir / "coverage_curve.png")

with tab_err:
    split = st.radio("Evaluation set", evals or ["test"], horizontal=True, key="err_split")
    show_image(rdir / f"eval_{split}" / "error_gallery.png")

with tab_data:
    eda = reports / "eda"
    show_image(eda / "class_counts.png")
    show_image(eda / "split_counts.png")
    show_image(eda / "samples.png")
