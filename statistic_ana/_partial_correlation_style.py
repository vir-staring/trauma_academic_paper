"""Shared, manuscript-ready plots for partial-correlation analyses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.fonttype"] = "none"


BLUE, NAVY, RED = "#3F75B5", "#24577E", "#B56B6B"
PALE_BLUE, GRAY, GRID = "#D7E4F2", "#666B73", "#E4E7EB"
SCATTER_POINT_SIZE = 24
LABELS = {
    "childhood_trauma": "Childhood trauma score",
    "ms_symbolic_score_grade_z": "Mathematical symbolic problem solving",
    "ms_verbalized_score_grade_z": "Mathematical verbalized problem solving",
    "ms_situational_score_grade_z": "Mathematical situational problem solving",
    "rit_corrected_fluency_grade_z": "Silent Sentence Reading",
    "rsw_ans_score_grade_z": "Character Recognition",
    "Orbito-Affective_FA_grade_z": "Orbito-affective network FA",
}


def value_text(value):
    return ("−" if value < 0 else "") + f"{abs(value):.3f}".lstrip("0")


def q_text(q):
    return "q < .001" if q < .001 else f"q = {q:.3f}".replace("0.", ".")


def stars(q):
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"


def style_axis(ax):
    ax.axhline(0, color=GRID, linewidth=.8)
    ax.axvline(0, color=GRID, linewidth=.8)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8A8A8A")


def plot_scatter(x_resid, y_resid, x, y, r, q, n, output_path):
    slope, intercept = np.polyfit(x_resid, y_resid, 1)
    x_line = np.linspace(x_resid.min(), x_resid.max(), 100)
    title = LABELS[y]
    y_label = title if "_FA_" in y else f"{title} score"
    if title.startswith("Mathematical"):
        y_label = y_label.replace(" problem solving", "\nproblem solving")
    fig, ax = plt.subplots(figsize=(10, 7.8))
    ax.scatter(x_resid, y_resid, s=SCATTER_POINT_SIZE, alpha=.28,
               color=BLUE, edgecolors="none")
    ax.plot(x_line, slope * x_line + intercept, color=RED, linewidth=2.8)
    style_axis(ax)
    ax.set_xlabel("Childhood trauma score", fontsize=20, labelpad=18)
    ax.set_ylabel(y_label, fontsize=20, labelpad=18)
    fig.suptitle(title, y=.95, fontsize=29, fontweight="bold")
    fig.text(.5, .875,
             f"partial r = {value_text(r)}; "
             f"{q_text(q)}; N = {n:,}", ha="center", fontsize=18,
             fontweight="normal", color=GRAY)
    fig.subplots_adjust(left=.15, right=.98, bottom=.16, top=.79)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_summary(results, output_path, title):
    labels = [LABELS[name] for name in results["y"]]
    values = results["partial_r"].to_numpy()
    y_pos = np.arange(len(results))
    height = max(4.2, 1.05 * len(results) + 2.2)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.axvline(0, color=GRID, linewidth=1)
    ax.hlines(y_pos, 0, values, color=PALE_BLUE, linewidth=3)
    ax.scatter(values, y_pos, s=120, color=BLUE, zorder=3)
    margin = max(.04, .35 * max(abs(values.min()), abs(values.max())))
    ax.set_xlim(min(values.min() - margin, -.02), max(values.max() + margin, .02))
    for index, row in results.reset_index(drop=True).iterrows():
        ax.text(.66, index,
                f"r = {value_text(row['partial_r'])}; {q_text(row['fdr_p'])} {stars(row['fdr_p'])}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=16,
                fontweight="bold" if row["fdr_p"] < .05 else "normal",
                color=NAVY if row["fdr_p"] < .05 else GRAY,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .4})
    ax.set_yticks(y_pos, labels, fontsize=17)
    ax.invert_yaxis()
    ax.set_xlabel("Partial correlation (r)", fontsize=19, labelpad=18)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#8A8A8A")
    fig.suptitle(title, y=.97, fontsize=28, fontweight="bold")
    fig.subplots_adjust(left=.40, right=.97, bottom=.17, top=.84)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
