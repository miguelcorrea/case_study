"""Figures for the chemical-versus-biological similarity analysis.

Every quantity plotted here is a property of a compound pair, not of a
compound.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from similarity_analysis import extract_upper_triangle

FIGURE_DPI = 250
MORPHOLOGY_COLOUR = "#2d6a6a"
EXPRESSION_COLOUR = "#b5502f"
NEUTRAL_COLOUR = "#8a8a8a"

DEFAULT_BIN_EDGES = (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def _style_axis(axis) -> None:
    """Apply a consistent minimal style."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)


def _group_pairs_into_chemical_bins(
    chemical_pairs: np.ndarray,
    biological_pairs: np.ndarray,
    bin_edges: tuple[float, ...],
) -> tuple[list[np.ndarray], list[str], list[int]]:
    """Split biological similarities according to the chemical similarity of the pair."""
    grouped, labels, counts = [], [], []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (chemical_pairs >= lower) & (chemical_pairs < upper)
        if not in_bin.any():
            continue
        grouped.append(biological_pairs[in_bin])
        labels.append(f"{lower:.1f}-{upper:.1f}")
        counts.append(int(in_bin.sum()))
    return grouped, labels, counts


def plot_similarity_distributions_by_chemical_bin(
    chemical_similarity: np.ndarray,
    morphology_similarity: np.ndarray,
    expression_similarity: np.ndarray,
    axis,
    bin_edges: tuple[float, ...] = DEFAULT_BIN_EDGES,
) -> None:
    """Distribution of pairwise biological similarity within bands of chemical similarity.

    Boxplots rather than bars of means: the bins differ in size by four orders of
    magnitude, and the distributions overlap heavily. Showing only the means
    would imply a cleaner separation than the data supports.

    Notches mark the 95% confidence interval of the median, so the width of the
    notch communicates how much the sparsely populated high-similarity bins
    should be trusted.
    """
    chemical_pairs = extract_upper_triangle(chemical_similarity)

    morphology_groups, labels, counts = _group_pairs_into_chemical_bins(
        chemical_pairs, extract_upper_triangle(morphology_similarity), bin_edges
    )
    expression_groups, _, _ = _group_pairs_into_chemical_bins(
        chemical_pairs, extract_upper_triangle(expression_similarity), bin_edges
    )

    positions = np.arange(len(labels))
    offset = 0.19
    box_width = 0.32

    for groups, position_offset, colour in (
        (morphology_groups, -offset, MORPHOLOGY_COLOUR),
        (expression_groups, +offset, EXPRESSION_COLOUR),
    ):
        axis.boxplot(
            groups,
            positions=positions + position_offset,
            widths=box_width,
            notch=True,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.4},
            boxprops={"facecolor": colour, "edgecolor": colour, "linewidth": 0.8},
            whiskerprops={"color": colour, "linewidth": 0.8},
            capprops={"color": colour, "linewidth": 0.8},
        )

    axis.axhline(0, color=NEUTRAL_COLOUR, linewidth=0.7, linestyle="--", zorder=0)

    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8.5)
    axis.set_xlim(-0.6, len(labels) - 0.4)

    # Pair counts go on a secondary axis so they cannot collide with the tick labels.
    count_axis = axis.secondary_xaxis("top")
    count_axis.set_xticks(positions)
    count_axis.set_xticklabels([f"{count:,}" for count in counts], fontsize=7)
    count_axis.tick_params(length=0)
    count_axis.set_xlabel("number of compound pairs in bin", fontsize=7.5, labelpad=4)
    count_axis.spines["top"].set_visible(False)

    axis.set_xlabel(
        "Chemical similarity between pairs of compounds", fontsize=9
    )
    axis.set_ylabel(
        "Biological similarity between pairs of compounds",
        fontsize=9,
    )
    axis.set_title(
        "Chemically similar compounds elicit more similar responses,\nbut only among close analogues",
        fontsize=10,
        pad=32,
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MORPHOLOGY_COLOUR, edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=EXPRESSION_COLOUR, edgecolor="none"),
    ]
    axis.legend(
        legend_handles,
        ["Morphology (Cell Painting)", "Expression (L1000)"],
        fontsize=8,
        frameon=False,
        loc="upper left",
    )
    _style_axis(axis)


def plot_modality_agreement(
    morphology_similarity: np.ndarray,
    expression_similarity: np.ndarray,
    axis,
    n_trend_bins: int = 18,
) -> None:
    """Agreement between the two assays, one hexagon per group of compound pairs.

    A running median is overlaid because at this density the scatter alone reads
    as a featureless blob: the correlation is real but weak, and the trend line
    is what makes it visible.
    """
    morphology_pairs = extract_upper_triangle(morphology_similarity)
    expression_pairs = extract_upper_triangle(expression_similarity)

    axis.hexbin(
        morphology_pairs,
        expression_pairs,
        gridsize=45,
        bins="log",
        cmap="Blues",
        linewidths=0,
    )
    axis.axhline(0, color=NEUTRAL_COLOUR, linewidth=0.6)
    axis.axvline(0, color=NEUTRAL_COLOUR, linewidth=0.6)

    # # Running median of transcriptomic similarity across quantiles of
    # # morphological similarity.
    # quantile_edges = np.quantile(morphology_pairs, np.linspace(0, 1, n_trend_bins + 1))
    # bin_centres, bin_medians = [], []
    # for lower, upper in zip(quantile_edges[:-1], quantile_edges[1:]):
    #     in_bin = (morphology_pairs >= lower) & (morphology_pairs < upper)
    #     if in_bin.sum() < 20:
    #         continue
    #     bin_centres.append(float(np.median(morphology_pairs[in_bin])))
    #     bin_medians.append(float(np.median(expression_pairs[in_bin])))

    # axis.plot(
    #     bin_centres,
    #     bin_medians,
    #     color=EXPRESSION_COLOUR,
    #     linewidth=1.8,
    #     marker="o",
    #     markersize=3,
    #     label="Running median",
    # )

    rho = spearmanr(morphology_pairs, expression_pairs).statistic
    axis.text(
        0.04,
        0.95,
        f"Spearman $\\rho$ = {rho:.3f}\n{len(morphology_pairs):,} compound pairs",
        transform=axis.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 3},
    )

    axis.set_xlabel(
        "Morphological similarity between compound pairs\n(cosine, Cell Painting)",
        fontsize=9,
    )
    axis.set_ylabel(
        "Transcriptomic similarity between compound pairs\n(cosine, L1000)", fontsize=9
    )
    axis.set_title(
        "L1000 and Cell Painting profiles correlate weakly", fontsize=10
    )
    axis.legend(fontsize=8, frameon=False, loc="lower right")
    _style_axis(axis)
    axis.grid(False)


def plot_mechanism_separation(mechanism_scores: dict[str, dict[str, float]], axis) -> None:
    """How well each similarity space separates compound pairs sharing a mechanism.

    Framed as a classification of pairs: given the similarity of two compounds,
    how well does it predict whether they share an annotated mechanism of action?
    An AUC of 0.5 is chance, marked with the dashed line.
    """
    labels = list(mechanism_scores)
    values = [mechanism_scores[label]["auc"] for label in labels]
    colours = [NEUTRAL_COLOUR, MORPHOLOGY_COLOUR, EXPRESSION_COLOUR][: len(labels)]

    axis.barh(np.arange(len(labels)), values, color=colours, height=0.55)
    axis.axvline(0.5, color="black", linewidth=0.9, linestyle="--")
    #axis.text(0.502, -0.45, "chance", fontsize=7.5, color="black", va="center")

    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels([label.replace(" (", "\n(") for label in labels], fontsize=8)
    axis.set_xlim(0.45, 0.80)
    axis.set_xlabel("AUC",fontsize=9,)
    axis.set_title(
        "Chemical structure separates mechanisms\nbetter than either biological assay", fontsize=10
    )
    for position, value in enumerate(values):
        axis.text(value + 0.005, position, f"{value:.3f}", va="center", fontsize=8.5)
    _style_axis(axis)
    axis.grid(axis="x", alpha=0.25, linewidth=0.6)


def build_summary_figure(
    chemical_similarity: np.ndarray,
    morphology_similarity: np.ndarray,
    expression_similarity: np.ndarray,
    mechanism_scores: dict[str, dict[str, float]],
    output_path: str | Path,
) -> Path:
    """Assemble the three panels into one figure and save it."""
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    plot_similarity_distributions_by_chemical_bin(
        chemical_similarity, morphology_similarity, expression_similarity, axes[0]
    )
    plot_modality_agreement(morphology_similarity, expression_similarity, axes[1])
    plot_mechanism_separation(mechanism_scores, axes[2])

    figure.tight_layout(w_pad=2.5)
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path