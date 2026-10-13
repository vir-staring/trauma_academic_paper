"""Compact forest plots for mediation analyses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.fonttype"] = "none"


BLUE, RED, GRID = "#3F75B5", "#B56B6B", "#E4E7EB"
OUTCOME_LABELS = {
    "ms_total_score_grade_z": "Mathematics",
    "cr_total_score_grade_z": "Chinese reading",
}


def forest_plot(labels, effects, lows, highs, colors, title, xlabel, output_path, figsize,
                xlabel_size=18):
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color="black", linewidth=1.4, zorder=2)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for position, effect, low, high, color in zip(y, effects, lows, highs, colors):
        ax.errorbar(effect, position, xerr=[[effect - low], [high - effect]], fmt="none",
                    ecolor=color, elinewidth=2.2, capsize=5, capthick=2, zorder=2)
    ax.scatter(effects, y, s=75, c=colors, zorder=3)
    ax.set_yticks(y, labels, fontsize=17)
    ax.invert_yaxis()
    ax.set_ylim(len(labels) - .55, -.55)
    ax.set_xlabel(xlabel, fontsize=xlabel_size, labelpad=14)
    ax.tick_params(axis="x", labelsize=15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#8A8A8A")
    span = max(highs.max() - lows.min(), .001)
    ax.set_xlim(lows.min() - .06 * span, highs.max() + .06 * span)
    fig.suptitle(title, y=.97, fontsize=21, fontweight="bold")
    fig.subplots_adjust(left=.31, right=.98, bottom=.27 if "\n" in xlabel else .20, top=.76)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def plot_simple_mediation(rows, outcomes, output_path):
    indexed = {outcome: row for outcome, (_, row) in zip(outcomes, rows.iterrows())}
    order = ["cr_total_score_grade_z", "ms_total_score_grade_z"]
    selected = [indexed[outcome] for outcome in order]
    forest_plot(
        [OUTCOME_LABELS[outcome] for outcome in order],
        np.array([row["indirect_effect"] for row in selected]),
        np.array([row["indirect_ci95_low"] for row in selected]),
        np.array([row["indirect_ci95_high"] for row in selected]),
        [RED if outcome == "cr_total_score_grade_z" else BLUE for outcome in order],
        "Simple mediation: bootstrap indirect effects",
        "Indirect effect through\norbito-affective network FA",
        output_path,
        (8.2, 4.0),
        xlabel_size=17,
    )


def plot_age_moderated_mediation(rows, output_path):
    order = ["cr_total_score_grade_z", "ms_total_score_grade_z"]
    age_order = ["mean_minus_sd", "mean", "mean_plus_sd"]
    selected, labels, colors = [], [], []
    for outcome in order:
        outcome_rows = rows[rows["y"] == outcome].set_index("age_point").loc[age_order]
        for _, row in outcome_rows.iterrows():
            selected.append(row)
            labels.append(f"{OUTCOME_LABELS[outcome]}: {row['age_months']:.1f} months")
            colors.append(RED if outcome == "cr_total_score_grade_z" else BLUE)
    forest_plot(
        labels,
        np.array([row["conditional_indirect"] for row in selected]),
        np.array([row["conditional_indirect_ci95_low"] for row in selected]),
        np.array([row["conditional_indirect_ci95_high"] for row in selected]),
        colors,
        "Age-moderated mediation: conditional indirect effects",
        "Conditional indirect effect",
        output_path,
        (9.0, 6.8),
    )
