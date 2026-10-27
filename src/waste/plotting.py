"""Shared matplotlib style for every report figure: one palette, recessive axes, thin marks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to reports/, never shown
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical order (assigned in fixed order, never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Single-hue sequential ramp (blue, light -> dark) for magnitudes such as confusion matrices.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "waste_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def apply_style() -> None:
    """Apply the project-wide matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "lines.markersize": 4,
            "font.family": "DejaVu Sans",
            "text.color": TEXT,
        }
    )


def save(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a figure tightly, create the folder and close the figure."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p
