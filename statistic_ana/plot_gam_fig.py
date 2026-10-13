#!/usr/bin/env python3
"""Plot the two GAM result panels as separate figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["svg.fonttype"] = "none"


# 直接修改这两个路径
TABLE_ROOT = Path("/path/to/output/gam")
OUTPUT_DIR = Path("/path/to/output/gam/figures")
RED, BLUE, GRAY, GRID = "#C44E52", "#376A9F", "#6B7280", "#E5E7EB"


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A3A3A3")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(labelsize=15)
    ax.set_axisbelow(True)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def p_text(value):
    return f"{value:.4f}".removeprefix("0")


def plot_group_curves(curves, tests, path):
    fig, ax = plt.subplots(figsize=(12.0, 6.3))
    for group_name, color, label in (
        ("Adverse", RED, "Trauma group"),
        ("Non-adverse", BLUE, "Normal group"),
    ):
        group = curves[curves["group"] == group_name].sort_values("age_years")
        x = group["age_years"].to_numpy()
        ax.plot(x, group["fitted_value"], color=color, linewidth=2.6, label=label)
        ax.fill_between(
            x, group["lower_95_ci"], group["upper_95_ci"], color=color, alpha=0.18
        )

    result = tests.iloc[0]
    ax.text(
        0.03,
        0.96,
        f"Shape difference: p = {p_text(result['轨迹形状差异_p'])}\n"
        f"Intercept difference: p = {p_text(result['截距差异_p'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
    )
    ax.set_title(
        "Cross-sectional age patterns of orbito-affective network FA",
        loc="left",
        fontsize=22,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Age (years)", fontsize=18, labelpad=12)
    ax.set_ylabel("orbito-affective network FA", fontsize=18, labelpad=12)
    ax.legend(frameon=False, fontsize=16, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    save(fig, path)


def plot_smooth_difference(data, path):
    data = data.sort_values("age_years")
    x = data["age_years"].to_numpy()
    estimate = data["smooth_difference"].to_numpy()
    lower = data["lower_95_ci"].to_numpy()
    upper = data["upper_95_ci"].to_numpy()
    significant = data["significant"].astype(str).str.lower().eq("true").to_numpy()

    fig, ax = plt.subplots(figsize=(12.0, 6.3))
    ax.fill_between(x, lower, upper, color="#B8BEC7", alpha=0.45)
    ax.fill_between(
        x,
        lower,
        upper,
        where=significant & (estimate > 0),
        color=RED,
        alpha=0.38,
        interpolate=False,
        label="Trauma group > normal group",
    )
    ax.fill_between(
        x,
        lower,
        upper,
        where=significant & (estimate < 0),
        color=BLUE,
        alpha=0.38,
        interpolate=False,
        label="Trauma group < normal group",
    )
    ax.plot(x, estimate, color=GRAY, linewidth=2.5)
    ax.axhline(0, color="#111827", linewidth=1.1, linestyle="--")
    ax.set_title(
        "Between-group difference in age-related smooths",
        loc="left",
        fontsize=22,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Age (years)", fontsize=18, labelpad=12)
    ax.set_ylabel(
        "Smooth difference\n(trauma group − normal group)", fontsize=18, labelpad=12
    )
    ax.legend(frameon=False, fontsize=16, loc="lower left")
    style_axis(ax)
    fig.tight_layout()
    save(fig, path)


def main():
    curves = pd.read_csv(TABLE_ROOT / "Orbito-Affective_FA_group_curve_pointwise_ci.csv")
    differences = pd.read_csv(
        TABLE_ROOT / "Orbito-Affective_FA_smooth_difference_pointwise_ci.csv"
    )
    tests = pd.read_csv(TABLE_ROOT / "GAM轨迹检验汇总.csv")

    plot_group_curves(curves, tests, OUTPUT_DIR / "figure_04a_gam_group_curves.svg")
    plot_smooth_difference(
        differences, OUTPUT_DIR / "figure_04b_gam_smooth_difference.svg"
    )
    print(f"Saved two GAM figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
