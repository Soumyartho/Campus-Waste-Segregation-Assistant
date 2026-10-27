"""Shared app helpers: cached resources, styling and the recommendation card."""

from __future__ import annotations

import hashlib
import html
import io
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # works even without 'pip install -e .'
    sys.path.insert(0, str(ROOT / "src"))

from waste.disposal import ClassAdvice, DisposalEngine, DisposalMapError, Recommendation  # noqa: E402
from waste.logstore import LogStore  # noqa: E402
from waste.predict import Prediction, Predictor, State  # noqa: E402
from waste.utils import class_names_from_map, load_yaml  # noqa: E402

# Overridable so tests (or a second deployment) can use their own model, thresholds and database.
INFERENCE_CONFIG = os.environ.get("WASTE_INFERENCE_CONFIG", "configs/inference.yaml")
ACCENT = "#2a78d6"  # single-hue bars: magnitude, not identity
STATE_STYLE = {
    State.CONFIDENT: ("Confident", "#1b7f4c", "#e7f5ec"),
    State.UNCERTAIN: ("Uncertain", "#8a5a00", "#fdf3dc"),
    State.UNRECOGNISED: ("Not recognised", "#a12d2d", "#fbe9e9"),
}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.wsa-card {border: 1px solid var(--wsa-border); border-radius: 14px; padding: 1.1rem 1.25rem;
           background: var(--wsa-surface); margin-bottom: 0.8rem;}
.wsa-pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
           font-weight: 600; letter-spacing: .02em;}
.wsa-class {font-size: 1.7rem; font-weight: 700; margin: .35rem 0 .1rem 0; line-height: 1.2;}
.wsa-muted {font-size: 0.9rem;}
.wsa-bin {display: flex; align-items: center; gap: .8rem; border-radius: 12px; padding: .7rem .9rem;
          margin: .8rem 0; border: 1px solid var(--wsa-border);}
.wsa-swatch {width: 42px; height: 42px; border-radius: 10px; flex: none;
             box-shadow: inset 0 0 0 2px rgba(255,255,255,.35);}
.wsa-bin-name {font-weight: 700; font-size: 1.05rem;}
.wsa-list {margin: .2rem 0 .6rem 1.1rem; padding: 0;}
.wsa-list li {margin: .15rem 0;}
.wsa-h {font-weight: 650; font-size: .92rem; margin-top: .55rem;}
.wsa-note {border-left: 3px solid #1baf7a; padding: .35rem .7rem; font-size: .9rem;
           margin-top: .5rem;}
.wsa-kpi {font-size: 1.9rem; font-weight: 700; line-height: 1.1;}
/* Translucent tokens so the same CSS works under Streamlit's light and dark themes. */
:root {--wsa-border: rgba(128,128,128,.28); --wsa-surface: rgba(128,128,128,.06);}
.wsa-muted, .wsa-note {opacity: .78;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def _load_predictor(config_path: str) -> Predictor | None:
    try:
        return Predictor.from_config(config_path)
    except FileNotFoundError:
        return None


def get_predictor() -> Predictor | None:
    """Load the model once per server process; None when the checkpoint is missing."""
    return _load_predictor(INFERENCE_CONFIG)


@st.cache_resource
def get_engine(class_names: tuple[str, ...]) -> DisposalEngine:
    """Validated disposal engine (raises DisposalMapError at startup if the map is incomplete)."""
    return DisposalEngine.from_config(INFERENCE_CONFIG, list(class_names))


@st.cache_resource
def _open_store(db_path: str) -> LogStore:
    return LogStore(db_path)


def get_store() -> LogStore:
    return _open_store(load_yaml(INFERENCE_CONFIG)["app"]["db_path"])


def app_config() -> dict:
    return load_yaml(INFERENCE_CONFIG)


def class_names() -> list[str]:
    """Class names from the loaded model, or from class_map.yaml when no model is loaded."""
    predictor = get_predictor()
    return predictor.class_names if predictor else class_names_from_map("configs/class_map.yaml")


def require_model() -> tuple[Predictor, DisposalEngine]:
    """Return (predictor, engine) or stop the page with a clear message."""
    predictor = get_predictor()
    if predictor is None:
        cfg = app_config()["model"]
        st.error(
            f"**No trained model found** at `{cfg['models_dir']}/{cfg['run_name']}/best.pt`.\n\n"
            "Train one with `make train` (or `./tasks.ps1 train` on Windows), or copy a "
            "`best.pt` into that folder, then reload this page."
        )
        st.stop()
    try:
        engine = get_engine(tuple(predictor.class_names))
    except DisposalMapError as exc:
        st.error(f"**Disposal map problem:** {exc}")
        st.stop()
    return predictor, engine


@st.cache_data(ttl=30)
def sample_images() -> dict[str, Path]:
    """Demo photos: everything in app/samples/, plus one held-out test image per class."""
    found = {p.stem: p for p in sorted((ROOT / "app" / "samples").glob("*"))
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    manifest = ROOT / load_yaml("configs/data.yaml")["paths"]["manifest"]
    if manifest.exists():
        test = pd.read_csv(manifest).query("split == 'test'").sort_values("path")
        for label, path in test.groupby("label")["path"].first().items():
            if (ROOT / path).exists():
                found.setdefault(f"test-{label}", ROOT / path)
    return found


def open_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an upright RGB image."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def image_key(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _e(text: str) -> str:
    return html.escape(str(text))


def state_pill(state: State) -> str:
    label, fg, bg = STATE_STYLE[state]
    return f'<span class="wsa-pill" style="color:{fg};background:{bg}">{label}</span>'


def bin_badge(a: ClassAdvice) -> str:
    recyclable = "Recyclable" if a.recyclable else "Not recyclable"
    return (
        f'<div class="wsa-bin"><div class="wsa-swatch" style="background:{_e(a.bin_color)}"></div>'
        f'<div><div class="wsa-bin-name">{_e(a.bin)} bin</div>'
        f'<div class="wsa-muted">{_e(a.bin_label)} · {recyclable}</div></div></div>'
    )


def _list(items: list[str], marker: str = "") -> str:
    return '<ul class="wsa-list">' + "".join(f"<li>{marker}{_e(i)}</li>" for i in items) + "</ul>"


def advice_html(a: ClassAdvice, compact: bool = False) -> str:
    """Bin badge, preparation steps, do-not list and impact note for one class."""
    parts = [bin_badge(a), '<div class="wsa-h">How to prepare it</div>', _list(a.steps)]
    parts += ['<div class="wsa-h">Do not</div>', _list(a.do_not)]
    if not compact:
        parts.append(f'<div class="wsa-note">{_e(a.impact_note)}</div>')
    return "".join(parts)


def render_recommendation(pred: Prediction, rec: Recommendation) -> None:
    """The main result card for one prediction."""
    if rec.state == State.CONFIDENT:
        a = rec.primary
        st.markdown(
            f'<div class="wsa-card">{state_pill(rec.state)}'
            f'<div class="wsa-class">{_e(a.display_name)}</div>'
            f'<div class="wsa-muted">Confidence {pred.confidence:.0%}</div></div>',
            unsafe_allow_html=True,
        )
        st.progress(min(pred.confidence, 1.0))
        st.markdown(f'<div class="wsa-card">{advice_html(a)}</div>', unsafe_allow_html=True)
        return

    st.markdown(
        f'<div class="wsa-card">{state_pill(rec.state)}'
        f'<div class="wsa-class">{_e(rec.title)}</div>'
        f'<div style="font-weight:600;margin:.3rem 0">{_e(rec.rule)}</div>'
        f"{_list(rec.advice)}</div>",
        unsafe_allow_html=True,
    )
    if rec.candidates:
        st.markdown("**Most likely categories**")
        cols = st.columns(len(rec.candidates))
        for col, a in zip(cols, rec.candidates, strict=True):
            with col:
                st.markdown(
                    f'<div class="wsa-card"><div class="wsa-class" style="font-size:1.2rem">'
                    f'{_e(a.display_name)}</div><div class="wsa-muted">{a.probability:.0%} likely'
                    f"</div>{advice_html(a, compact=True)}</div>",
                    unsafe_allow_html=True,
                )


def text_color() -> str:
    """Primary text ink for chart labels in the active Streamlit theme."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    return "#f0efec" if dark else "#0b0b0b"


def probability_chart(pred: Prediction, display: dict[str, str]) -> alt.Chart:
    """Horizontal bars of the top-k class probabilities, with value labels and tooltips."""
    df = pd.DataFrame(
        [{"class": display.get(c, c), "probability": p} for c, p in pred.top_k]
    )
    base = alt.Chart(df).encode(
        y=alt.Y("class:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                title=None),
        tooltip=[alt.Tooltip("class:N", title="Class"),
                 alt.Tooltip("probability:Q", format=".1%", title="Probability")],
    )
    bars = base.mark_bar(color=ACCENT, cornerRadiusEnd=4, height=18)
    labels = base.mark_text(align="left", dx=4, fontSize=12, color=text_color()).encode(
        text=alt.Text("probability:Q", format=".0%")
    )
    return (bars + labels).properties(height=40 * len(df) + 10)


def display_names(engine: DisposalEngine) -> dict[str, str]:
    return {c: v["display_name"] for c, v in engine.classes.items()}


def bar_chart(df: pd.DataFrame, cat: str, val: str, title: str, fmt: str = "d") -> alt.Chart:
    """Sorted single-hue horizontal bar chart with tooltips."""
    base = alt.Chart(df).encode(
        y=alt.Y(f"{cat}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X(f"{val}:Q", title=None),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q", format=fmt)],
    )
    bars = base.mark_bar(color=ACCENT, cornerRadiusEnd=4)
    labels = base.mark_text(align="left", dx=4, fontSize=11, color=text_color()).encode(
        text=alt.Text(f"{val}:Q", format=fmt)
    )
    return (bars + labels).properties(title=title, height=max(28 * len(df), 80))
