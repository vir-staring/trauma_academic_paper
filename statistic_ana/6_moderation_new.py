#!/usr/bin/env python3
"""Moderation models with simple slopes and Johnson-Neyman plots."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/path/to/output/mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.multitest import multipletests

plt.rcParams["svg.fonttype"] = "none"


DATA_TABLE = Path("/path/to/data/final/env_bhv_cov.csv")
SUBJECT_ID = "subject_id"

X_COLUMNS = ["childhood_trauma"]
Y_COLUMNS = ["ms_total_score_grade_z", "cr_total_score_grade_z"]
W_COLUMNS = ["ssrt_reaction_grade_z"]
COVARIATES = ["gender", "f_ses", ]

DO_FDR = True
RESULT_DIR = Path("/path/to/output/moderate")
FIGURE_DIR = Path("/path/to/output/moderate/moderatemoderation_jn")

OUTCOME_LABELS = {
    "ms_total_score_grade_z": ("Mathematics", "mathematics"),
    "cr_total_score_grade_z": ("Chinese reading", "reading"),
}
COLORS = {"low": "#3F75B5", "mean": "#7A7F87", "high": "#D35D60"}


def safe_name(text):
    for old in ["/", "\\", " ", ":", "*", "?", "\"", "<", ">", "|"]:
        text = text.replace(old, "_")
    return text


def make_model_data(data, x, y, w, covariates):
    model_data = data[[x, y, w] + covariates].copy()
    text_covariates = [c for c in covariates if model_data[c].dtype == "object"]
    model_data = pd.get_dummies(model_data, columns=text_covariates, drop_first=True)
    model_data = model_data.apply(pd.to_numeric, errors="coerce").dropna()
    covariate_columns = [c for c in model_data.columns if c not in [x, y, w]]

    # 中心化 X 和 W，交互项含义更清楚；协变量保持原始编码进入模型。
    model_data["x_c"] = model_data[x] - model_data[x].mean()
    model_data["w_c"] = model_data[w] - model_data[w].mean()
    model_data["xw"] = model_data["x_c"] * model_data["w_c"]
    return model_data, covariate_columns


def fit_model(model_data, y, covariate_columns):
    x_terms = ["x_c", "w_c", "xw"] + covariate_columns
    return sm.OLS(model_data[y], sm.add_constant(model_data[x_terms], has_constant="add")).fit()


def slope_at(model, w_c):
    beta = model.params["x_c"] + model.params["xw"] * w_c
    cov = model.cov_params()
    se = np.sqrt(cov.loc["x_c", "x_c"] + w_c ** 2 * cov.loc["xw", "xw"] + 2 * w_c * cov.loc["x_c", "xw"])
    t_value = beta / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_value), model.df_resid))
    ci_low = beta - stats.t.ppf(0.975, model.df_resid) * se
    ci_high = beta + stats.t.ppf(0.975, model.df_resid) * se
    return beta, se, t_value, p_value, ci_low, ci_high


def johnson_neyman_regions(model, moderator_values, alpha=0.05):
    w_values = np.asarray(moderator_values, dtype=float)
    w_mean = w_values.mean()
    w_min = w_values.min()
    w_max = w_values.max()

    beta_x = model.params["x_c"]
    beta_xw = model.params["xw"]
    cov = model.cov_params()
    var_x = cov.loc["x_c", "x_c"]
    var_xw = cov.loc["xw", "xw"]
    cov_x_xw = cov.loc["x_c", "xw"]
    t_critical = stats.t.ppf(1 - alpha / 2, model.df_resid)

    # Solve effect(W)^2 - t_critical^2 * Var[effect(W)] = 0.
    quadratic = beta_xw ** 2 - t_critical ** 2 * var_xw
    linear = 2 * (beta_x * beta_xw - t_critical ** 2 * cov_x_xw)
    constant = beta_x ** 2 - t_critical ** 2 * var_x
    scale = max(abs(quadratic), abs(linear), abs(constant), 1.0)
    tolerance = 1e-12 * scale

    roots_centered = []
    solution_status = "no_real_critical_point"
    if abs(quadratic) <= tolerance:
        if abs(linear) > tolerance:
            roots_centered = [-constant / linear]
            solution_status = "one_real_critical_point"
        elif abs(constant) <= tolerance:
            solution_status = "all_values_on_significance_boundary"
    else:
        discriminant = linear ** 2 - 4 * quadratic * constant
        discriminant_tolerance = 1e-12 * max(
            linear ** 2,
            abs(4 * quadratic * constant),
            1.0,
        )
        if discriminant > discriminant_tolerance:
            sqrt_discriminant = np.sqrt(discriminant)
            roots_centered = [
                (-linear - sqrt_discriminant) / (2 * quadratic),
                (-linear + sqrt_discriminant) / (2 * quadratic),
            ]
            solution_status = "two_real_critical_points"
        elif abs(discriminant) <= discriminant_tolerance:
            roots_centered = [-linear / (2 * quadratic)]
            solution_status = "one_repeated_critical_point"

    roots_centered = sorted(set(float(root) for root in roots_centered))
    roots_raw = [root + w_mean for root in roots_centered]

    critical_points = []
    if roots_raw:
        for point_id, (w_c, w_raw) in enumerate(zip(roots_centered, roots_raw), start=1):
            beta, se, t_value, p_value, ci_low, ci_high = slope_at(model, w_c)
            within_range = w_min <= w_raw <= w_max
            critical_points.append({
                "solution_status": solution_status,
                "critical_point_id": point_id,
                "critical_value": w_raw,
                "critical_value_centered": w_c,
                "within_observed_range": within_range,
                "conditional_effect": beta,
                "se": se,
                "t": t_value,
                "p": p_value,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })
    else:
        critical_points.append({
            "solution_status": solution_status,
            "critical_point_id": pd.NA,
            "critical_value": np.nan,
            "critical_value_centered": np.nan,
            "within_observed_range": False,
            "conditional_effect": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        })

    boundary_tolerance = 1e-10 * max(abs(w_min), abs(w_max), 1.0)
    interior_roots = [
        root for root in roots_raw
        if w_min + boundary_tolerance < root < w_max - boundary_tolerance
    ]
    boundaries = [w_min] + interior_roots + [w_max]
    intervals = []
    for region_id, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        representative = (lower + upper) / 2
        beta, se, t_value, p_value, ci_low, ci_high = slope_at(
            model, representative - w_mean
        )
        is_significant = p_value < alpha
        if is_significant and beta > 0:
            direction = "Positive"
        elif is_significant and beta < 0:
            direction = "Negative"
        else:
            direction = "Not significant"

        if region_id < len(boundaries) - 1:
            observed_mask = (w_values >= lower) & (w_values < upper)
        else:
            observed_mask = (w_values >= lower) & (w_values <= upper)

        intervals.append({
            "region_id": region_id,
            "interval_lower": lower,
            "interval_upper": upper,
            "lower_boundary_type": (
                "observed_min" if np.isclose(lower, w_min) else "JN_critical_point"
            ),
            "upper_boundary_type": (
                "observed_max" if np.isclose(upper, w_max) else "JN_critical_point"
            ),
            "significant_0.05": is_significant,
            "effect_direction": direction,
            "representative_value": representative,
            "representative_effect": beta,
            "representative_se": se,
            "representative_t": t_value,
            "representative_p": p_value,
            "representative_ci95_low": ci_low,
            "representative_ci95_high": ci_high,
            "n_observed_in_interval": int(observed_mask.sum()),
            "percent_observed_in_interval": 100 * observed_mask.mean(),
            "extends_to_observed_min": np.isclose(lower, w_min),
            "extends_to_observed_max": np.isclose(upper, w_max),
        })

    metadata = {
        "alpha": alpha,
        "t_critical": t_critical,
        "moderator_mean": w_mean,
        "moderator_sd": w_values.std(ddof=1),
        "observed_min": w_min,
        "observed_max": w_max,
    }
    return critical_points, intervals, metadata


def value_text(value, digits=3):
    text = f"{abs(value):.{digits}f}".lstrip("0")
    return ("−" if value < 0 else "") + text


def significance_mark(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def style_axis(ax):
    ax.axhline(0, color="#E4E7EB", linewidth=.8)
    ax.grid(axis="y", color="#E4E7EB", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8A8A8A")


def plot_moderation(model_id, y, w, model, model_data, w_mean, w_sd,
                    jn_results, jn_critical_results, output_path):
    title, outcome = OUTCOME_LABELS[y]
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.8))
    x_grid = np.linspace(model_data["x_c"].min(), model_data["x_c"].max(), 100)
    covariates = [name for name in model.params.index
                  if name not in ["const", "x_c", "w_c", "xw"]]
    baseline = model.params["const"] + sum(
        model.params[name] * model_data[name].mean() for name in covariates
    )
    levels = [
        ("low", "Shorter SSRT (−1 SD)", w_mean - w_sd),
        ("mean", "Mean SSRT", w_mean),
        ("high", "Longer SSRT (+1 SD)", w_mean + w_sd),
    ]

    for row, (key, label, w_raw) in enumerate(levels):
        w_c = w_raw - w_mean
        slope, _, _, p_value, _, _ = slope_at(model, w_c)
        prediction = baseline + model.params["w_c"] * w_c + slope * x_grid
        axes[0].plot(x_grid, prediction, color=COLORS[key], linewidth=2.8)
        axes[0].text(.77, .91 - row * .11,
                     f"b = {value_text(slope)}{significance_mark(p_value)}",
                     transform=axes[0].transAxes, color=COLORS[key], fontsize=19,
                     fontweight="bold", ha="left")

    axes[0].text(.5, 1.46, "Simple slopes at three SSRT levels",
                 transform=axes[0].transAxes, ha="center", fontsize=23, fontweight="bold")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=10,
                      markerfacecolor=COLORS[key], markeredgecolor="none", label=label)
               for key, label, _ in levels]
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.24), ncol=3,
                   frameon=False, fontsize=15, handlelength=1, columnspacing=1.8)
    axes[0].text(-.03, 1.08, f"Predicted {outcome} score",
                 transform=axes[0].transAxes, fontsize=18, color="#666B73")
    axes[0].set_xlabel("Childhood trauma", fontsize=19, labelpad=20)
    style_axis(axes[0])

    one_jn = jn_results[jn_results["model_id"] == model_id]
    moderator = one_jn["moderator_value"].to_numpy(dtype=float)
    effect = one_jn["conditional_effect"].to_numpy(dtype=float)
    ci_low = one_jn["ci95_low"].to_numpy(dtype=float)
    ci_high = one_jn["ci95_high"].to_numpy(dtype=float)
    axes[1].fill_between(moderator, ci_low, ci_high, color="#D7E4F2")
    axes[1].plot(moderator, effect, color="#24577E", linewidth=2.8)

    critical = jn_critical_results[
        (jn_critical_results["model_id"] == model_id)
        & (jn_critical_results["within_observed_range"])
    ]["critical_value"].dropna().to_numpy(dtype=float)
    x_span = moderator.max() - moderator.min()
    for index, threshold in enumerate(critical):
        axes[1].axvline(threshold, color="#D35D60", linestyle=(0, (6, 5)), linewidth=1.8)
        label_x = threshold + .045 * x_span
        axes[1].text(label_x, .94 - index * .10,
                     f"JN threshold = {value_text(threshold, 2)}",
                     transform=axes[1].get_xaxis_transform(), color="#8C3030",
                     fontsize=18, va="top")

    axes[1].text(.5, 1.32, f"Conditional trauma–{outcome} association",
                 transform=axes[1].transAxes, ha="center", fontsize=23, fontweight="bold")
    axes[1].text(-.03, 1.08, "Conditional effect (b) with 95% CI",
                 transform=axes[1].transAxes, fontsize=18, color="#666B73")
    axes[1].set_xlabel("SSRT", fontsize=19, labelpad=20)
    style_axis(axes[1])

    fig.suptitle(title, y=.985, fontsize=31, fontweight="bold")
    fig.text(.98, .025, "*** p < .001    ** p < .01    * p < .05",
             ha="right", fontsize=14, color="#666B73")
    fig.subplots_adjust(left=.07, right=.98, bottom=.14, top=.64, wspace=.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def make_demo_data(n=1600):
    rng = np.random.default_rng(20260908)
    trauma = np.clip(rng.normal(1.45, .40, n), .8, 3.3)
    ssrt = rng.normal(0, 1, n)
    gender = rng.binomial(1, .47, n)
    ses = rng.normal(0, 1, n)
    x_c, w_c = trauma - trauma.mean(), ssrt - ssrt.mean()
    common = .08 * gender + .16 * ses - .18 * w_c
    return pd.DataFrame({
        "childhood_trauma": trauma,
        "ssrt_reaction_grade_z": ssrt,
        "gender": gender,
        "f_ses": ses,
        "ms_total_score_grade_z": common - .38 * x_c - .04 * x_c * w_c + rng.normal(0, 1, n),
        "cr_total_score_grade_z": common - .62 * x_c - .30 * x_c * w_c + rng.normal(0, 1, n),
    })


DEMO = "--demo" in sys.argv[1:]
if DEMO:
    RESULT_DIR = Path("/path/to/output/moderation_demo/tables")
    FIGURE_DIR = Path("/path/to/output/moderation_demo")
    data = make_demo_data()
else:
    data = pd.read_csv(DATA_TABLE)
model_rows = []
coef_rows = []
simple_rows = []
jn_rows = []
jn_critical_rows = []
jn_interval_rows = []
plot_items = []

for x in X_COLUMNS:
    for y in Y_COLUMNS:
        for w in W_COLUMNS:
            model_data, covariate_columns = make_model_data(data, x, y, w, COVARIATES)
            model = fit_model(model_data, y, covariate_columns)
            model_id = safe_name(x + "__" + y + "__" + w)

            # 主要检验项是 X*W，即环境变量与调节变量的交互效应。
            model_rows.append({
                "model_id": model_id,
                "x": x,
                "y": y,
                "moderator": w,
                "covariates": "; ".join(covariate_columns),
                "n": int(model.nobs),
                "df_resid": model.df_resid,
                "r2": model.rsquared,
                "adj_r2": model.rsquared_adj,
                "f": model.fvalue,
                "f_p": model.f_pvalue,
                "interaction_beta": model.params["xw"],
                "interaction_se": model.bse["xw"],
                "interaction_t": model.tvalues["xw"],
                "interaction_p": model.pvalues["xw"],
            })

            for term in model.params.index:
                ci_low, ci_high = model.conf_int().loc[term]
                coef_rows.append({
                    "model_id": model_id,
                    "term": term,
                    "beta": model.params[term],
                    "se": model.bse[term],
                    "t": model.tvalues[term],
                    "p": model.pvalues[term],
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                })

            w_mean = model_data[w].mean()
            w_sd = model_data[w].std()

            critical_points, intervals, jn_metadata = johnson_neyman_regions(
                model, model_data[w]
            )
            jn_base = {
                "model_id": model_id,
                "x": x,
                "y": y,
                "moderator": w,
                "n": int(model.nobs),
                "interaction_beta": model.params["xw"],
                "interaction_se": model.bse["xw"],
                "interaction_p": model.pvalues["xw"],
                **jn_metadata,
            }
            for critical_point in critical_points:
                jn_critical_rows.append({**jn_base, **critical_point})
            for interval in intervals:
                jn_interval_rows.append({**jn_base, **interval})

            observed_critical_values = [
                point["critical_value"]
                for point in critical_points
                if point["within_observed_range"]
            ]
            significant_intervals = [
                interval for interval in intervals if interval["significant_0.05"]
            ]
            model_rows[-1]["jn_critical_values_within_observed_range"] = (
                "; ".join(f"{value:.6g}" for value in observed_critical_values)
                if observed_critical_values
                else "No critical point within observed range"
            )
            model_rows[-1]["jn_critical_point_count_within_observed_range"] = len(
                observed_critical_values
            )
            model_rows[-1]["jn_significant_intervals_within_observed_range"] = (
                "; ".join(
                    f"{interval['interval_lower']:.6g} to "
                    f"{interval['interval_upper']:.6g} "
                    f"({interval['effect_direction']})"
                    for interval in significant_intervals
                )
                if significant_intervals
                else "No significant interval within observed range"
            )
            model_rows[-1]["jn_significant_interval_count_within_observed_range"] = len(
                significant_intervals
            )

            for label, w_raw in {"low": w_mean - w_sd, "mean": w_mean, "high": w_mean + w_sd}.items():
                w_c = w_raw - w_mean
                beta, se, t_value, p_value, ci_low, ci_high = slope_at(model, w_c)
                simple_rows.append({
                    "model_id": model_id,
                    "moderator_level": label,
                    "moderator_value": w_raw,
                    "simple_slope": beta,
                    "se": se,
                    "t": t_value,
                    "p": p_value,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                })

            # Johnson-Neyman: scan the observed moderator range for conditional X effects.
            w_grid = np.linspace(model_data[w].min(), model_data[w].max(), 200)
            for w_raw in w_grid:
                beta, se, t_value, p_value, ci_low, ci_high = slope_at(model, w_raw - w_mean)
                jn_rows.append({
                    "model_id": model_id,
                    "moderator_value": w_raw,
                    "conditional_effect": beta,
                    "se": se,
                    "p": p_value,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "significant_0.05": p_value < 0.05,
                })
            plot_items.append((model_id, x, y, w, model, model_data, covariate_columns, w_mean, w_sd))

model_results = pd.DataFrame(model_rows)
coef_results = pd.DataFrame(coef_rows)
simple_results = pd.DataFrame(simple_rows)
jn_results = pd.DataFrame(jn_rows)
jn_critical_results = pd.DataFrame(jn_critical_rows)
jn_interval_results = pd.DataFrame(jn_interval_rows)

if DO_FDR:
    model_results["interaction_fdr_p"] = multipletests(model_results["interaction_p"], method="fdr_bh")[1]
    model_results["interaction_fdr_reject_0.05"] = multipletests(model_results["interaction_p"], method="fdr_bh")[0]

RESULT_DIR.mkdir(parents=True, exist_ok=True)
model_results.to_csv(RESULT_DIR / "moderation_model_summary.csv", index=False, na_rep="")
coef_results.to_csv(RESULT_DIR / "moderation_coefficients.csv", index=False, na_rep="")
simple_results.to_csv(RESULT_DIR / "moderation_simple_slopes.csv", index=False, na_rep="")
jn_results.to_csv(RESULT_DIR / "moderation_johnson_neyman.csv", index=False, na_rep="")
jn_critical_results.to_csv(
    RESULT_DIR / "moderation_johnson_neyman_critical_points.csv",
    index=False,
    na_rep="",
)
jn_interval_results.to_csv(
    RESULT_DIR / "moderation_johnson_neyman_intervals.csv",
    index=False,
    na_rep="",
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
for model_id, x, y, w, model, model_data, covariate_columns, w_mean, w_sd in plot_items:
    plot_moderation(
        model_id, y, w, model, model_data, w_mean, w_sd,
        jn_results, jn_critical_results,
        FIGURE_DIR / (model_id + ".svg"),
    )

print("Model summary:", RESULT_DIR / "moderation_model_summary.csv")
print("Coefficients:", RESULT_DIR / "moderation_coefficients.csv")
print("Simple slopes:", RESULT_DIR / "moderation_simple_slopes.csv")
print("Johnson-Neyman:", RESULT_DIR / "moderation_johnson_neyman.csv")
print(
    "JN critical points:",
    RESULT_DIR / "moderation_johnson_neyman_critical_points.csv",
)
print(
    "JN intervals:",
    RESULT_DIR / "moderation_johnson_neyman_intervals.csv",
)
print("Figures:", FIGURE_DIR)
