#!/usr/bin/env python3
"""Compare matched trauma groups by school stage and draw violin plots."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/path/to/output/mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

plt.rcParams["svg.fonttype"] = "none"


GROUP_DIR = Path("/path/to/data/final/psm_trauma_results")
ADVERSE_FILE = GROUP_DIR / "不利.csv"
NON_ADVERSE_FILE = GROUP_DIR / "无不利.csv"
ENVBHV_FILE = Path("/path/to/data/final/env_bhv_cov.csv")
RESULT_TABLE = Path("/path/to/output/ttest/stage_group_difference.csv")
FIGURE_DIR = RESULT_TABLE.parent

SUBJECT_ID, GRADE_COLUMN = "subject_id", "grade"
METRIC_LABELS = {
    "ms_symbolic_score_grade_z": "Mathematical symbolic problem solving",
    "ms_verbalized_score_grade_z": "Mathematical verbalized problem solving",
    "ms_situational_score_grade_z": "Mathematical situational problem solving",
    "rsw_ans_score_grade_z": "Character Recognition",
    "rit_corrected_fluency_grade_z": "Silent Sentence Reading",
}
STAGES = {
    "G1-3": [1, 2, 3],
    "G4-6": [4, 5, 6],
    "G7-9": [7, 8, 9],
    "G10-12": [10, 11, 12],
}
STAGE_LABELS = ["Grades 1–3", "Grades 4–6", "Grades 7–9", "Grades 10–12"]
GROUPS = ["adverse", "non_adverse"]
COLORS = {"adverse": "#D35D60", "non_adverse": "#3F75B5"}
OFFSETS = {"adverse": -0.15, "non_adverse": 0.15}


def read_table(path):
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")


def read_ids(path, group):
    frame = read_table(path)
    frame = frame.rename(columns={frame.columns[0]: SUBJECT_ID})
    frame[SUBJECT_ID] = frame[SUBJECT_ID].astype(str).str.strip()
    return frame[[SUBJECT_ID]].assign(group=group)


def load_data():
    groups = pd.concat([
        read_ids(ADVERSE_FILE, "adverse"),
        read_ids(NON_ADVERSE_FILE, "non_adverse"),
    ], ignore_index=True)
    data = read_table(ENVBHV_FILE)
    data = data.rename(columns={data.columns[0]: SUBJECT_ID})
    data[SUBJECT_ID] = data[SUBJECT_ID].astype(str).str.strip()
    data[GRADE_COLUMN] = pd.to_numeric(data[GRADE_COLUMN], errors="coerce")
    missing = [column for column in METRIC_LABELS if column not in data]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")
    data[list(METRIC_LABELS)] = data[list(METRIC_LABELS)].apply(pd.to_numeric, errors="coerce")
    return data.merge(groups, on=SUBJECT_ID, how="inner")


def compare_stages(data, metric):
    rows, values = [], []
    for stage, grades in STAGES.items():
        subset = data[data[GRADE_COLUMN].isin(grades)].dropna(subset=[metric, "group"])
        samples = {group: subset.loc[subset["group"] == group, metric] for group in GROUPS}
        if any(sample.empty for sample in samples.values()):
            raise ValueError(f"{metric}, {stage}: one group has no valid observations")
        adverse, normal = samples.values()
        u_value, p_value = mannwhitneyu(adverse, normal, alternative="two-sided")
        rows.append({
            "stage": stage,
            "grades": ",".join(map(str, grades)),
            "metric": metric,
            "n_adverse": len(adverse),
            "n_non_adverse": len(normal),
            "mean_adverse": adverse.mean(),
            "mean_non_adverse": normal.mean(),
            "sd_adverse": adverse.std(),
            "sd_non_adverse": normal.std(),
            "median_adverse": adverse.median(),
            "median_non_adverse": normal.median(),
            "iqr_adverse": adverse.quantile(.75) - adverse.quantile(.25),
            "iqr_non_adverse": normal.quantile(.75) - normal.quantile(.25),
            "mannwhitney_u": u_value,
            "p": p_value,
            "rank_biserial": 1 - 2 * u_value / (len(adverse) * len(normal)),
        })
        values.extend({"stage": stage, "group": group, "value": value}
                      for group, sample in samples.items() for value in sample)

    results = pd.DataFrame(rows)
    results["fdr_p"] = multipletests(results["p"], method="fdr_bh")[1]
    results["fdr_reject_0.05"] = results["fdr_p"] < .05
    return results, pd.DataFrame(values)


def significance_mark(q):
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"


def draw_figure(values, results, metric, output_path):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    rng = np.random.default_rng(123)
    y_min, y_max = values["value"].min(), values["value"].max()
    span = max(y_max - y_min, 1)
    bracket_y, bracket_h = y_max + .08 * span, .025 * span

    for index, stage in enumerate(STAGES):
        for group in GROUPS:
            sample = values.loc[(values["stage"] == stage) & (values["group"] == group), "value"].to_numpy()
            x = index + OFFSETS[group]
            violin = ax.violinplot(sample, [x], widths=.28, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[group])
                body.set_edgecolor("none")
                body.set_alpha(.24)
            ax.scatter(x + rng.uniform(-.045, .045, len(sample)), sample, s=18,
                       color=COLORS[group], alpha=.62, edgecolors="none", zorder=3)
            ax.hlines(sample.mean(), x - .10, x + .10, color="white", linewidth=5, zorder=4)
            ax.hlines(sample.mean(), x - .10, x + .10, color=COLORS[group], linewidth=2.6, zorder=5)

        left, right = index + OFFSETS["adverse"], index + OFFSETS["non_adverse"]
        ax.plot([left, left, right, right],
                [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                color="#252525", linewidth=1.4, clip_on=False)
        mark = significance_mark(results.loc[results["stage"] == stage, "fdr_p"].iat[0])
        ax.text(index, bracket_y + bracket_h + .025 * span, mark, ha="center", va="bottom",
                fontsize=22, fontweight="bold" if mark != "ns" else "normal")

    ax.axhline(0, color="#E4E7EB", linewidth=.8)
    ax.set_xlim(-.55, len(STAGES) - .45)
    ax.set_ylim(y_min - .10 * span, bracket_y + .16 * span)
    ax.set_xticks(range(len(STAGES)), STAGE_LABELS, fontsize=17)
    ax.tick_params(axis="x", pad=12)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("School stage", fontsize=20, labelpad=20)
    y_label = f"{METRIC_LABELS[metric]} score"
    if y_label.startswith("Mathematical"):
        y_label = y_label.replace(" problem solving", "\nproblem solving")
    ax.set_ylabel(y_label, fontsize=20, labelpad=14)
    ax.grid(axis="y", color="#E4E7EB", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8A8A8A")

    handles = [Line2D([], [], marker="o", linestyle="", markersize=11,
                      markerfacecolor=COLORS[group], markeredgecolor="none", label=label)
               for group, label in zip(GROUPS, ["Trauma group", "Normal group"])]
    fig.suptitle(METRIC_LABELS[metric], y=.965, fontsize=28, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .89), ncol=2,
               frameon=False, fontsize=20, handletextpad=.7, columnspacing=4)
    fig.text(.98, .035, "*** q < .001    ** q < .01    * q < .05    ns q ≥ .05",
             ha="right", fontsize=14, color="#666666")
    fig.subplots_adjust(left=.10, right=.98, bottom=.17, top=.75)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def demo_data():
    rng = np.random.default_rng(20260908)
    means = [(-.75, .10), (-.68, .08), (-.62, .10), (-.08, -.05)]
    rows = []
    for grades, (adverse_mean, normal_mean) in zip(STAGES.values(), means):
        for group, mean in zip(GROUPS, [adverse_mean, normal_mean]):
            rows.extend({GRADE_COLUMN: grades[0], "group": group,
                         "rit_corrected_fluency_grade_z": value}
                        for value in rng.normal(mean, .78, 60))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="render a fake-data preview")
    parser.add_argument("--demo-output", type=Path,
                        default=Path("/path/to/output/stage_group_difference_demo.svg"))
    args = parser.parse_args()

    if args.demo:
        metric = "rit_corrected_fluency_grade_z"
        results, values = compare_stages(demo_data(), metric)
        draw_figure(values, results, metric, args.demo_output)
        print(results[["stage", "fdr_p", "fdr_reject_0.05"]].to_string(index=False))
        print("Demo figure:", args.demo_output)
        return

    data, all_results = load_data(), []
    for metric in METRIC_LABELS:
        results, values = compare_stages(data, metric)
        all_results.append(results)
        draw_figure(values, results, metric, FIGURE_DIR / f"{metric}.svg")
    RESULT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_results, ignore_index=True).to_csv(RESULT_TABLE, index=False, na_rep="")
    print("Results:", RESULT_TABLE)


if __name__ == "__main__":
    main()
