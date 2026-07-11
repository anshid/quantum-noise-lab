"""Publication-quality plotting helpers for measurement outcomes.

Color and layout choices follow a fixed, colorblind-validated categorical
palette rather than matplotlib defaults: series 1 (blue #2a78d6) for empirical/
measured data, series 2 (aqua #1baf7a) for theoretical/reference data, applied
consistently across the project so a color always means the same thing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

_SURFACE = "#fcfcfb"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_MUTED = "#898781"
_PRIMARY_INK = "#0b0b0b"
_SERIES_EMPIRICAL = "#2a78d6"
_SERIES_THEORETICAL = "#1baf7a"


def _style_axes(ax: matplotlib.axes.Axes) -> None:
    ax.set_facecolor(_SURFACE)
    ax.figure.set_facecolor(_SURFACE)
    ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(_AXIS)
    ax.tick_params(colors=_MUTED)
    ax.title.set_color(_PRIMARY_INK)


def plot_counts_histogram(
    counts: dict[str, int],
    *,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
    save_path: Path | None = None,
) -> matplotlib.axes.Axes:
    """Bar chart of raw measurement counts, one bar per observed bitstring."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    labels = sorted(counts)
    values = [counts[label] for label in labels]

    bars = ax.bar(labels, values, color=_SERIES_EMPIRICAL, width=0.6, zorder=2)
    ax.bar_label(bars, padding=3, color=_PRIMARY_INK, fontsize=9)
    ax.set_xlabel("Measured bitstring")
    ax.set_ylabel("Counts")
    ax.set_title(title)
    _style_axes(ax)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_probability_comparison(
    empirical: dict[str, float],
    theoretical: dict[str, float],
    *,
    title: str = "",
    save_path: Path | None = None,
) -> matplotlib.axes.Axes:
    """Grouped bar chart comparing empirical (shot-based) vs theoretical (Born-rule) probabilities."""
    labels = sorted(set(empirical) | set(theoretical))
    x = np.arange(len(labels))
    width = 0.35

    _, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(
        x - width / 2,
        [empirical.get(label, 0.0) for label in labels],
        width,
        label="Empirical (shots)",
        color=_SERIES_EMPIRICAL,
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        [theoretical.get(label, 0.0) for label in labels],
        width,
        label="Theoretical (Born rule)",
        color=_SERIES_THEORETICAL,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Measured bitstring")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend(frameon=False, labelcolor=_PRIMARY_INK)
    _style_axes(ax)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
